"""AIR pipeline, extracted unchanged from the experiment notebooks so scripts can reuse it.

chunk_document -> distill_chunk_adapter (per-chunk Micro-LoRA context distillation) -> belief states
-> confidence_gate / relevance gate -> max_cross_jsd.  icl_baseline is the split-window ICL baseline.
"""
from itertools import combinations

import torch
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model

CHUNK_SIZE = 256
CHUNK_OVERLAP = 64
TOP_K = 3
DISTILL_STEPS = 10
DISTILL_LR = 1e-3
MAX_GRAD_NORM = 1.0


def lora_config(r=2, alpha=None):
    return LoraConfig(
        r=r, lora_alpha=alpha if alpha is not None else 2 * r,
        target_modules=["up_proj", "down_proj", "gate_proj", "q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.0, bias="none", task_type="CAUSAL_LM",
    )


def answer_token_ids(tok):
    return [tok.encode("Yes", add_special_tokens=False)[0], tok.encode("No", add_special_tokens=False)[0]]


def chunk_document(text, tok, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    token_ids = tok.encode(text, add_special_tokens=False)
    stride = chunk_size - overlap
    chunks, start = [], 0
    while start < len(token_ids):
        end = min(start + chunk_size, len(token_ids))
        chunks.append(tok.decode(token_ids[start:end], skip_special_tokens=True))
        if end >= len(token_ids):
            break
        start += stride
    return chunks


def format_prompt(doc, query):
    return (
        "Answer the question with a single word: Yes or No. No filler words.\n\n"
        "Document: The human body has 206 bones in the adult skeleton.\n"
        "Question: Is this claim true or false: Do adults have more than 200 bones?\n"
        "Answer: Yes\n\n"
        f"Document: {doc}\nQuestion: {query}\nAnswer:"
    )


def distill_chunk_adapter(model, chunk_text, query, tok, cfg, answer_ids,
                          steps=DISTILL_STEPS, lr=DISTILL_LR, max_grad_norm=MAX_GRAD_NORM):
    """Context-distil one chunk into a fresh LoRA; return (belief over [Yes, No], confidence, unloaded model)."""
    peft_model = get_peft_model(model, cfg)
    params = [p for p in peft_model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr)
    peft_model.train()
    t_ids = tok(format_prompt(chunk_text, query), return_tensors="pt", truncation=True, max_length=512).input_ids.to(peft_model.device)
    s_ids = tok(format_prompt("", query), return_tensors="pt", truncation=True, max_length=128).input_ids.to(peft_model.device)
    for _ in range(steps):
        with peft_model.disable_adapter(), torch.no_grad():
            t_dist = torch.softmax(peft_model(t_ids).logits[:, -1, :].float(), dim=-1)
        s_log = torch.log_softmax(peft_model(s_ids).logits[:, -1, :].float(), dim=-1)
        loss = F.kl_div(s_log, t_dist, reduction="batchmean", log_target=False)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, max_grad_norm)
        opt.step()
    peft_model.eval()
    with torch.no_grad():
        belief = torch.softmax(peft_model(s_ids).logits[:, -1, answer_ids].float(), dim=-1).squeeze(0).cpu()
    conf = belief.max().item()
    model = peft_model.unload()
    del peft_model, opt, params
    torch.cuda.empty_cache()
    return belief, conf, model


def compute_jsd(p, q, eps=1e-10):
    p, q = p.clamp(min=eps), q.clamp(min=eps)
    m = 0.5 * (p + q)
    return (0.5 * (p * (p / m).log()).sum() + 0.5 * (q * (q / m).log()).sum()).item()


def confidence_gate(beliefs, confs, top_k=TOP_K):
    if len(beliefs) <= top_k:
        return beliefs, list(range(len(beliefs)))
    idx = sorted(range(len(confs)), key=lambda i: confs[i], reverse=True)[:top_k]
    return [beliefs[i] for i in idx], idx


def max_cross_jsd(states, confs=None):
    if len(states) < 2:
        return 0.0
    best = 0.0
    for i, j in combinations(range(len(states)), 2):
        d = compute_jsd(states[i], states[j])
        if confs is not None:
            d *= confs[i] * confs[j]
        best = max(best, d)
    return best


def air_beliefs(document, query, model, tok, cfg, answer_ids, steps=DISTILL_STEPS):
    """Distil every chunk; return per-chunk beliefs, confidences, chunk texts, and the (unloaded) model."""
    chunks = chunk_document(document, tok)
    beliefs, confs = [], []
    for ch in chunks:
        b, c, model = distill_chunk_adapter(model, ch, query, tok, cfg, answer_ids, steps=steps)
        beliefs.append(b)
        confs.append(c)
    return beliefs, confs, chunks, model


def icl_split_window(document, query, model, tok, answer_ids, eps=1e-10, max_length=2048):
    ids = tok.encode(document, add_special_tokens=False)
    mid = len(ids) // 2
    halves = [tok.decode(ids[:mid], skip_special_tokens=True), tok.decode(ids[mid:], skip_special_tokens=True)]
    dists = []
    with torch.no_grad():
        for h in halves:
            x = tok(format_prompt(h, query), return_tensors="pt", truncation=True, max_length=max_length).input_ids.to(model.device)
            dists.append(torch.softmax(model(x).logits[:, -1, answer_ids].float(), dim=-1).squeeze(0).cpu().clamp(min=eps))
    return compute_jsd(dists[0], dists[1])
