import json, time
import numpy as np
from huggingface_hub import snapshot_download
from safetensors import safe_open
from transformers import AutoTokenizer

# ---------- settings ----------
MODEL = "HuggingFaceTB/SmolLM2-135M"
PROMPT = "The capital of France is"
N_NEW = 500
TEMPERATURE, TOP_K, TOP_P, REP_PENALTY = 0.7, 40, 0.9, 1.15
MAX_SEQ = 512

# ---------- load model ----------
path = snapshot_download(MODEL)
cfg = json.load(open(f"{path}/config.json"))
tok = AutoTokenizer.from_pretrained(path)

W = {}
with safe_open(f"{path}/model.safetensors", framework="pt") as f:
    for name in f.keys():
        W[name] = f.get_tensor(name).float().numpy()      # stored as bf16, we use fp32

n_layers = cfg["num_hidden_layers"]
n_heads  = cfg["num_attention_heads"]
n_kv     = cfg["num_key_value_heads"]
dim      = cfg["hidden_size"]
hd       = dim // n_heads                                  # size of ONE head
eps      = cfg["rms_norm_eps"]
theta    = cfg["rope_theta"]

k_cache = np.zeros((n_layers, MAX_SEQ, n_kv, hd), dtype=np.float32)
v_cache = np.zeros((n_layers, MAX_SEQ, n_kv, hd), dtype=np.float32)

# ---------- building blocks ----------
def rmsnorm(x, w, eps):
    return x / np.sqrt(np.mean(x * x) + eps) * w

def softmax(x):
    e = np.exp(x - x.max())
    return e / e.sum()

def silu(x):
    return x / (1.0 + np.exp(-x))

def rope(x, pos, theta):                                   # x: (n_heads, head_dim)
    d = x.shape[-1]; half = d // 2
    inv_freq = 1.0 / (theta ** (np.arange(half) * 2.0 / d))
    ang = pos * inv_freq
    cos, sin = np.cos(ang), np.sin(ang)
    x1, x2 = x[:, :half], x[:, half:]
    return np.concatenate([x1 * cos - x2 * sin, x2 * cos + x1 * sin], axis=-1)

# ---------- the model: one token in, scores for every vocabulary word out ----------
def forward(token, pos):
    x = W["model.embed_tokens.weight"][token]
    for l in range(n_layers):
        p = f"model.layers.{l}."
        # attention
        h = rmsnorm(x, W[p + "input_layernorm.weight"], eps)
        q = (W[p + "self_attn.q_proj.weight"] @ h).reshape(n_heads, hd)
        k = (W[p + "self_attn.k_proj.weight"] @ h).reshape(n_kv, hd)
        v = (W[p + "self_attn.v_proj.weight"] @ h).reshape(n_kv, hd)
        q, k = rope(q, pos, theta), rope(k, pos, theta)
        k_cache[l, pos], v_cache[l, pos] = k, v
        out = np.zeros((n_heads, hd), dtype=np.float32)
        for head in range(n_heads):
            kvh = head // (n_heads // n_kv)                # GQA: heads share K/V
            K = k_cache[l, :pos + 1, kvh]
            V = v_cache[l, :pos + 1, kvh]
            scores = (K @ q[head]) / np.sqrt(hd)
            out[head] = softmax(scores) @ V
        x = x + W[p + "self_attn.o_proj.weight"] @ out.reshape(-1)
        # MLP
        h = rmsnorm(x, W[p + "post_attention_layernorm.weight"], eps)
        gate = W[p + "mlp.gate_proj.weight"] @ h
        up   = W[p + "mlp.up_proj.weight"] @ h
        x = x + W[p + "mlp.down_proj.weight"] @ (silu(gate) * up)
    x = rmsnorm(x, W["model.norm.weight"], eps)
    head_w = W.get("lm_head.weight", W["model.embed_tokens.weight"])
    return head_w @ x

# ---------- choosing the next token ----------
rng = np.random.default_rng()

def apply_rep_penalty(logits, seen_ids, penalty):
    for t in set(seen_ids):                                # make already-used tokens less attractive
        logits[t] = logits[t] / penalty if logits[t] > 0 else logits[t] * penalty
    return logits

def sample(logits, temperature, top_k, top_p):
    if temperature == 0:
        return int(logits.argmax())                        # greedy
    logits = logits / temperature                          # lower = safer, higher = more random
    kth = np.sort(logits)[-top_k]
    logits = np.where(logits < kth, -np.inf, logits)       # top-k: keep only the best k tokens
    probs = softmax(logits)
    order = np.argsort(-probs)
    cum = np.cumsum(probs[order])
    keep = order[: np.searchsorted(cum, top_p) + 1]        # top-p: smallest set covering p of the probability
    p = probs[keep] / probs[keep].sum()
    return int(rng.choice(keep, p=p))

def generate(prompt_ids, n_new):
    assert len(prompt_ids) + n_new <= MAX_SEQ, "raise MAX_SEQ"
    ids = list(prompt_ids)
    t0 = time.perf_counter()
    for pos, t in enumerate(prompt_ids):                   # prefill
        logits = forward(t, pos)
    t1 = time.perf_counter()
    generated = 0
    for _ in range(n_new):                                 # decode
        logits = apply_rep_penalty(logits.copy(), ids, REP_PENALTY)
        nxt = sample(logits, TEMPERATURE, TOP_K, TOP_P)
        if nxt == tok.eos_token_id:
            break
        ids.append(nxt)
        generated += 1
        logits = forward(nxt, len(ids) - 1)
    t2 = time.perf_counter()
    print(f"prefill {len(prompt_ids)/(t1-t0):.1f} tok/s | decode {generated/(t2-t1):.1f} tok/s")
    return ids

if __name__ == "__main__":
    ids = generate(tok(PROMPT).input_ids, N_NEW)
    print(tok.decode(ids))