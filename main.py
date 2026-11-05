import json, numpy as np
from huggingface_hub import snapshot_download
from safetensors import safe_open

path = snapshot_download("HuggingFaceTB/SmolLM2-135M")
cfg = json.load(open(f"{path}/config.json"))
# print(cfg)  # read this: layers, heads, kv heads, hidden size, rope_theta, eps

W = {}
with safe_open(f"{path}/model.safetensors", framework="pt") as f:
    for name in f.keys():
        W[name] = f.get_tensor(name).float().numpy()      # stored as bf16, we use fp32

# for name in list[W][0]:
#     print(name, W[name].shape)

def rmsnorm(x, w, eps):
    return x/ np.sqrt(np.mean(x * x) + eps) * w

def softmax(x):
    e = np.exp(x - x.max())     # subtracting max avoids overflow, same result
    return e/e.sum()

def silu(x):
    return x/(1.0 + np.exp(-x))

def rope(x, pos, theta):    # x: (n_heads, head_dims)
    d = x.shape[-1]; half = d//2
    inv_freq = 1.0 / (theta ** (np.arange(half) * 2.0 / d))
    ang = pos * inv_freq
    cos, sin = np.cos(ang), np.sin(ang)
    x1, x2 = x[:, :half], x[:, half:]
    return np.concatenate([x1 * cos - x2* sin, x2 * cos + x1 * sin], axis=-1)


n_layers = cfg["num_hidden_layers"]
n_heads = cfg['num_attention_heads']
n_kv = cfg['num_key_value_heads']
dim = cfg['hidden_size']
hd = dim // n_heads
eps = cfg['rms_norm_eps']
theta = cfg['rope_theta']
MAX_SEQ =512

k_cache = np.zeros((n_layers, MAX_SEQ, n_kv, hd), dtype=np.float32)
v_cache = np.zeros((n_layers, MAX_SEQ, n_kv, hd), dtype=np.float32)

def forward(token, pos):
    x = W["model.embed_tokens.weight"][token]       # (dim,)
    for l in range(n_layers):
        p = f"model.layers.{l}."
        # attention
        h = rmsnorm(x, W[p+"input_layernorm.weight"], eps)
        q = (W[p+"self_attn.q_proj.weight"] @ h).reshape(n_heads, hd)
        k = (W[p+"self_attn.k_proj.weight"] @ h).reshape(n_kv, hd)
        v = (W[p+"self_attn.v_proj.weight"] @ h).reshape(n_kv, hd)
        q, k = rope(q, pos, theta), rope(k, pos, theta)
        k_cache[l, pos], v_cache[l, pos] = k,v
        out = np.zeros((n_heads, hd), dtype=np.float32)
        for head in range(n_heads):
            kvh = head // (n_heads // n_kv)     # GQA: heads share K/V

            # 1. Slice cached keys and values up to the current position (pos + 1 tokens)
            K = k_cache[l, :pos+1, kvh]     # (pos + 1, hd)
            V = v_cache[l, :pos+1, kvh]     # (pos + 1, hd)

            # 2. Scaled dot-product attention scores
            scores = (K @ q[head]) / np.sqrt(hd)       # (pos + 1,)

            # 3. Softmax to get attention probabilities
            weights = softmax(scores)  # (pos + 1,)

            # 4. Weighted combination of values
            out[head] = weights @ V  # (hd,)

        x = x + W[p+"self_attn.o_proj.weight"] @ out.reshape(-1)

        # MLP
        h = rmsnorm(x, W[p+"post_attention_layernorm.weight"], eps)
        gate_w = W[p+"mlp.gate_proj.weight"]
        up_w = W[p+"mlp.up_proj.weight"]
        down_w = W[p+"mlp.down_proj.weight"]

        x = x + down_w @ (silu(gate_w @ h) * (up_w @ h))

    x = rmsnorm(x, W['model.norm.weight'], eps)
    head_w = W.get('lm_head.weight', W['model.embed_tokens.weight'])
    return head_w @ x       # logits, (vocab,)

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

tok = AutoTokenizer.from_pretrained(path)
ref = AutoModelForCausalLM.from_pretrained(path, dtype=torch.float32)

def generate(prompt_ids, n_new):
    ids = list(prompt_ids)
    for pos, t in enumerate(prompt_ids):        # prefill: feed the prompt one token at a time
        logits = forward(t, pos)
    for _ in range(n_new):      # decode: one new token per step
        nxt = int(logits.argmax())      # greedy = always take the top score
        ids.append(nxt)
        logits = forward(nxt, len(ids)-1)
    return ids

import time

def timed_generate(prompt_ids, n_new):
    ids = list(prompt_ids)
    t0 = time.perf_counter()
    for pos, t in enumerate(prompt_ids):
        logits = forward(t, pos)
    t1 = time.perf_counter()
    for _ in range(n_new):
        nxt = int(logits.argmax())
        ids.append(nxt)
        logits = forward(nxt, len(ids) - 1)
    t2 = time.perf_counter()
    print(f"prefill: {len(prompt_ids)/(t1-t0):.1f} tok/s | decode: {n_new/(t2-t1):.1f} tok/s")
    return ids


prompt = 'The capital of France is Paris. One thing I like about Paris is'
ids = tok(prompt).input_ids

# mine = generate(ids, 30)
mine = timed_generate(ids, 300)    # run twice, ignore the first (warm-up)
theirs = ref.generate(torch.tensor([ids]), max_new_tokens = 300, do_sample=False, pad_token_id=tok.eos_token_id)[0].tolist()

print(mine== theirs)
print(tok.decode(mine))
