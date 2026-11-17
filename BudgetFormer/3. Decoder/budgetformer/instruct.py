import math
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from torch.optim import AdamW
from datasets import  DatasetDict
import torch.nn as nn
import numpy as np
from transformers.models.gpt2.modeling_gpt2 import GPT2MLP
from nltk.translate.bleu_score import sentence_bleu
from rouge_score import rouge_scorer
from bert_score import score as bertscore

# =========================
# CONFIG
# =========================
device = "cuda" if torch.cuda.is_available() else "cpu"
batch_size = 16
max_len = 1024
epochs = 2
lr = 5e-4
D_MODEL = 768
NUM_HEADS = 12
NUM_BLOCKS = 12

# Budget interval
S_MIN = 0.1
S_MAX = 0.9

# Adaptive alpha
ALPHA_BASE = 0.001
ALPHA_MAX = 0.05

BETA_MAX = 0.05
SIGMA_MAX = 0.5
TAU_MAX = 2.0
TAU_MIN = 0.1
GAMMA = 5.0

# =========================
# Schedules
# =========================
def tau_schedule(step, total_steps):
    return TAU_MIN + (TAU_MAX - TAU_MIN) * math.exp(-GAMMA * step / total_steps)

def beta_schedule(step, total_steps):
    return BETA_MAX * (1 - 2*step / total_steps)

def sigma_schedule(step, total_steps):
    return SIGMA_MAX * (1 - step / total_steps)

# =========================
# Adaptive alpha function
# =========================
def adaptive_alpha(s, s_min, s_max, alpha_base=ALPHA_BASE, alpha_max=ALPHA_MAX):
    over = torch.clamp(s - s_max, min=0)
    under = torch.clamp(s_min - s, min=0)
    violation = over + under
    return torch.clamp(alpha_base + violation, max=alpha_max)

# =========================
# Budget loss with s_min / s_max
# =========================
def budget_loss(ss):
    loss = 0
    for s in ss:
        alpha = adaptive_alpha(s, S_MIN, S_MAX)
        low = F.relu(S_MIN - s)
        high = F.relu(s - S_MAX)
        loss += (alpha * (low**2 + high**2)).mean()
    return loss


# =========================
# METRICS: FLOPS / MEMORY / CARBON
# =========================
def compute_flops(ss, seq_len, batch_size, d_model, num_heads, training=True):
    flops = 0
    d_head = d_model // num_heads

    for s in ss:
        if training:
            H_active = num_heads
        else:
            H_active = s.mean().item() * num_heads

        # =========================
        # 1. QKV projections
        # =========================
        flops_qkv = 3 * batch_size * seq_len * d_model * d_model

        # =========================
        # 2. QK^T
        # =========================
        flops_qk = batch_size * H_active * (seq_len**2) * d_head

        # =========================
        # 3. Softmax
        # =========================
        flops_softmax = batch_size * H_active * (seq_len**2)

        # =========================
        # 4. AV
        # =========================
        flops_av = batch_size * H_active * (seq_len**2) * d_head

        # =========================
        # 5. Gating
        # =========================
        flops_gating = batch_size * seq_len * H_active * d_head

        # =========================
        # 6. Output projection
        # =========================
        flops_out = batch_size * seq_len * d_model * d_model

        # =========================
        # 7. Budget + gate net
        # =========================
        flops_budget = batch_size * d_model * d_model
        flops_gate = batch_size * d_model * num_heads

        flops += (
            flops_qkv
            + flops_qk
            + flops_softmax
            + flops_av
            + flops_gating
            + flops_out
            + flops_budget
            + flops_gate
        )

    return flops

def compute_weight_memory(model):
    total_bytes = 0
    for p in model.parameters():
        total_bytes += p.numel() * p.element_size()
    return total_bytes / (1024**2)  # MB

def compute_activation_memory(ids, ss, d_model, num_heads, training=True):
    B, N = ids.shape
    d_head = d_model // num_heads
    bytes_per_float = 4

    total_mem = 0

    for s in ss:
        if training:
            H_active = num_heads
        else:
            H_active = s.mean().item() * num_heads

        # =========================
        # 1. Input
        # =========================
        mem_input = B * N * d_model

        # =========================
        # 2. QKV
        # =========================
        mem_qkv = 3 * B * N * d_model

        # =========================
        # 3. Attention scores
        # =========================
        mem_scores = B * H_active * (N * N)

        # =========================
        # 4. Softmax
        # =========================
        mem_attn = B * H_active * (N * N)

        # =========================
        # 5. Head outputs
        # =========================
        mem_heads = B * N * H_active * d_head

        # =========================
        # 6. Gating
        # =========================
        mem_gating = B * H_active

        # =========================
        # 7. Output
        # =========================
        mem_out = B * N * d_model

        mem_total_layer = (
            mem_input
            + mem_qkv
            + mem_scores
            + mem_attn
            + mem_heads
            + mem_gating
            + mem_out
        )

        # =========================
        # 🔥 Backward double approx
        # =========================
        if training:
            mem_total_layer *= 2  # approx gradients

        total_mem += mem_total_layer

    # bytes → MB
    return total_mem * bytes_per_float / (1024**2)

def compute_carbon(flops, gpu_tflops=10, power_watts=300, carbon_intensity=0.4):
    """
    312: A100 fp16
    10: A100 fp32
    Approximation basée sur throughput GPU
    """
    # temps ≈ FLOPs / (FLOPs/sec)
    time_seconds = flops / (gpu_tflops * 1e12)

    energy_kwh = (power_watts * time_seconds) / 1000 / 3600
    co2 = energy_kwh * carbon_intensity

    return co2 * 1000 #gco2

# ============================
# MODEL
# ============================
class AdaptiveAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_head = d_model // num_heads

        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out = nn.Linear(d_model, d_model)

        # Budget predictor
        self.budget_net = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, 1)
        )

        # Head importance
        self.gate_net = nn.Linear(d_model, num_heads)

    def forward(
        self,
        X,
        attention_mask=None,
        causal=False,
        tau=1.0,
        sigma=0.0,
        s_override=None,
        topk_only=False,
        past_k=None,
        past_v=None,
        use_cache=False
    ):
        """
        X: (B, N, D)
        past_k, past_v: (B, T_past, H, d_head)
        """

        B, N, D = X.shape

        # =========================
        # Budget + gating
        # =========================
        pooled = X.mean(dim=1)

        s_pred = torch.sigmoid(self.budget_net(pooled)).squeeze(-1)
        s_used = s_override.expand(B) if s_override is not None else s_pred

        logits = self.gate_net(pooled)
        if sigma > 0:
            logits = logits + torch.randn_like(logits) * sigma

        q_head = F.softmax(logits / tau, dim=-1)

        # Gating
        if topk_only:
            k_head = torch.clamp((s_used * self.num_heads).long(), min=1)  # (B,)

            k_max = k_head.max()

            topk_vals, topk_idx = torch.topk(q_head, k_max, dim=-1)  # (B, k_max)

            # =========================
            # Mask sans boucle
            # =========================
            range_k = torch.arange(k_max, device=X.device).unsqueeze(0)  # (1, k_max)
            mask = range_k < k_head.unsqueeze(1)  # (B, k_max)

            # scatter → mask_heads
            mask_heads = torch.zeros_like(q_head)  # (B, H)
            mask_heads.scatter_(1, topk_idx, mask.float())

            g = s_used.unsqueeze(-1) * self.num_heads * q_head * mask_heads

        else:
            g = s_used.unsqueeze(-1) * self.num_heads * q_head

        # =========================
        # QKV projection
        # =========================
        qkv = self.qkv(X).view(B, N, 3, self.num_heads, self.d_head)

        q_ = qkv[:, :, 0]  # (B, N, H, d_head)
        k_ = qkv[:, :, 1]
        v_ = qkv[:, :, 2]

        # =========================
        # KV CACHE
        # =========================
        if past_k is not None and past_v is not None:
            k_ = torch.cat([past_k, k_], dim=1)
            v_ = torch.cat([past_v, v_], dim=1)

        new_past_k = k_
        new_past_v = v_

        T_total = k_.size(1)

        # =========================
        # Attention scores
        # =========================
        scores = torch.einsum("bnhd,bmhd->bhnm", q_, k_) / math.sqrt(self.d_head)

        # =========================
        # Padding mask
        # =========================
        if attention_mask is not None:
            # attention_mask: (B, T_total)
            mask = attention_mask.unsqueeze(1).unsqueeze(2)
            scores = scores.masked_fill(mask == 0, float('-inf'))

        # =========================
        # Causal mask (ONLY if no cache)
        # =========================
        if causal and past_k is None:
            causal_mask = torch.tril(torch.ones(N, T_total, device=X.device))
            scores = scores.masked_fill(causal_mask == 0, float('-inf'))

        # =========================
        # Softmax
        # =========================
        attn = F.softmax(scores, dim=-1)

        # =========================
        # Weighted sum
        # =========================
        head_out = torch.einsum("bhnm,bmhd->bnhd", attn, v_)

        # =========================
        # Apply gating
        # =========================
        g_expand = g.unsqueeze(1).unsqueeze(-1)
        weighted = head_out * g_expand

        # =========================
        # Output projection
        # =========================
        out = self.out(weighted.reshape(B, N, -1))

        # =========================
        # Return
        # =========================
        if use_cache:
            return out, q_head, s_used, new_past_k, new_past_v
        else:
            return out, q_head, s_used, None, None

class MyGPT2Attention(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.attn = AdaptiveAttention(
            d_model=config.hidden_size,
            num_heads=config.num_attention_heads
        )

        self.resid_dropout = nn.Dropout(config.resid_pdrop)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        causal=False,
        tau=1.0,
        sigma=0.0,
        s_override=None,
        topk_only=False,
        past_k=None,
        past_v=None,
        use_cache=False
    ):
        attn_out, q_head, s_used, new_k, new_v = self.attn(
            hidden_states,
            attention_mask=attention_mask,
            causal=causal,
            tau=tau,
            sigma=sigma,
            s_override=s_override,
            topk_only=topk_only,
            past_k=past_k,
            past_v=past_v,
            use_cache=use_cache
        )

        attn_out = self.resid_dropout(attn_out)

        return attn_out, new_k, new_v, q_head, s_used

class MyGPT2Block(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.ln_1 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_epsilon)
        self.attn = MyGPT2Attention(config)

        self.ln_2 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_epsilon)
        self.mlp = GPT2MLP(config.n_embd, config)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        causal=False,
        tau=1.0,
        sigma=0.0,
        s_override=None,
        topk_only=False,
        past_k=None,
        past_v=None,
        use_cache=False
    ):
        # ===== Attention =====
        normed = self.ln_1(hidden_states)

        attn_out, new_k, new_v, q_head, s_used = self.attn(
            normed,
            attention_mask=attention_mask,
            causal=causal,
            tau=tau,
            sigma=sigma,
            s_override=s_override,
            topk_only=topk_only,
            past_k=past_k,
            past_v=past_v,
            use_cache=use_cache
        )

        hidden_states = hidden_states + attn_out

        # ===== MLP =====
        normed = self.ln_2(hidden_states)
        mlp_out = self.mlp(normed)

        hidden_states = hidden_states + mlp_out

        return hidden_states, new_k, new_v, q_head, s_used

class MyGPT2Model(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.wte = nn.Embedding(config.vocab_size, config.hidden_size)
        self.wpe = nn.Embedding(config.max_position_embeddings, config.hidden_size)

        self.drop = nn.Dropout(config.embd_pdrop)

        self.h = nn.ModuleList([
            MyGPT2Block(config) for _ in range(config.num_hidden_layers)
        ])

        self.ln_f = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_epsilon)

    def forward(
        self,
        input_ids,
        attention_mask=None,
        past_kvs=None, # list of tuples [(key, value), ...]*N_layers
        causal=False,
        use_cache=False,
        tau=1.0,
        sigma=0.0,
        s_override=None,
        topk_only=False
    ):
        B, T = input_ids.shape

        pos = torch.arange(T, device=input_ids.device).unsqueeze(0)

        x = self.wte(input_ids) + self.wpe(pos)
        x = self.drop(x)

        new_past = []
        all_q, all_s = [], []

        for i, block in enumerate(self.h):

            past_k, past_v = (None, None)
            if past_kvs is not None:
                past_k, past_v = past_kvs[i]

            x, new_k, new_v, q_head, s_used = block(
                x,
                attention_mask=attention_mask,
                causal=causal,
                tau=tau,
                sigma=sigma,
                s_override=s_override,
                topk_only=topk_only,
                past_k=past_k,
                past_v=past_v,
                use_cache=use_cache
            )

            if use_cache:
                new_past.append((new_k, new_v))

            all_q.append(q_head)
            all_s.append(s_used)

        x = self.ln_f(x)

        return x, new_past, all_q, all_s

class MyGPT2LM(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.transformer = MyGPT2Model(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def forward(
        self,
        input_ids,
        attention_mask=None,
        past_kvs=None, # list of tuples [(key, value), ...]*N_layers
        causal=False,
        use_cache=False,
        tau=1.0,
        sigma=0.0,
        s_override=None,
        topk_only=False
    ):
        hidden_states, new_past, all_q, all_s = self.transformer(
            input_ids,
            attention_mask=attention_mask,
            past_kvs=past_kvs,
            causal=causal,
            tau=tau,
            sigma=sigma,
            s_override=s_override,
            topk_only=topk_only,
            use_cache=use_cache
        )

        logits = self.lm_head(hidden_states)

        return {
            "logits": logits,
            "past_key_values": new_past,
            "q_heads": all_q,
            "s_used": all_s
        }

# =========================
# FUNCTIONS
# =========================
def format_example(ex):
    instruction = ex["instruction"]
    inp = ex["input"].strip() if ex["input"] and ex["input"].strip() != "" else "<noinput>"
    output = ex["output"]

    return {
        "instruction": instruction,
        "input": inp,
        "output": output
    }

def tokenize(ex, tokenizer, eos_id):

    inst_text = f"Instruction: {ex['instruction']}\n"
    input_text = f"Input: {ex['input']}\nResponse:"
    output_text = ex["output"]

    inst_ids = tokenizer(inst_text, add_special_tokens=False)["input_ids"]
    input_ids_part = tokenizer(input_text, add_special_tokens=False)["input_ids"]
    response_ids = tokenizer(output_text, add_special_tokens=False)["input_ids"]

    # 🔥 garantir EOS à la fin de la réponse
    if len(response_ids) == 0 or response_ids[-1] != eos_id:
        response_ids = response_ids + [eos_id]

    total_len = len(inst_ids) + len(input_ids_part) + len(response_ids)

    # =========================
    # CASE 1 : pas de truncation
    # =========================
    if total_len <= max_len:
        input_ids = inst_ids + input_ids_part + response_ids
        labels = [-100] * (len(inst_ids) + len(input_ids_part)) + response_ids

    # =========================
    # CASE 2 : truncation nécessaire
    # =========================
    else:
        remaining = max_len - len(inst_ids) - len(response_ids)

        # -------------------------
        # CASE 2A : réponse trop longue (pas de place pour input)
        # -------------------------
        if remaining <= 0:
            trunc_len = max_len - len(inst_ids)

            response_trunc = response_ids[:trunc_len]

            # 🔥 garantir EOS même après truncation
            if response_trunc[-1] != eos_id:
                response_trunc[-1] = eos_id

            input_ids = inst_ids + response_trunc
            labels = [-100] * len(inst_ids) + response_trunc

        # -------------------------
        # CASE 2B : input partiel + réponse complète
        # -------------------------
        else:
            input_ids_part = input_ids_part[-remaining:]

            input_ids = inst_ids + input_ids_part + response_ids
            labels = [-100] * (len(inst_ids) + len(input_ids_part)) + response_ids

    return {
        "input_ids": input_ids[:max_len],
        "labels": labels[:max_len]
    }

def collate_fn(batch, eos_token_id):

    input_ids = torch.nn.utils.rnn.pad_sequence(
        [torch.tensor(x["input_ids"]) for x in batch],
        batch_first=True,
        padding_value=eos_token_id
    )

    labels = torch.nn.utils.rnn.pad_sequence(
        [torch.tensor(x["labels"]) for x in batch],
        batch_first=True,
        padding_value=-100
    )

    return {
        "input_ids": input_ids,
        "labels": labels
    }

def compute_loss(logits, labels):

    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()

    return F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1)
    )

def evaluate_ppl(model, val_loader, device, tau=1.0, top_k=False, training=True):

    model.eval()

    total_loss = 0.0
    n_batches = 0
    val_flops, val_carbon, val_mem, val_peak_mem_sum = 0, 0, 0, 0

    with torch.no_grad():
        all_s_used = []
        for batch in val_loader:

            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            torch.cuda.reset_peak_memory_stats(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, tau=tau, sigma=0.0, top_k=top_k)
            v_peak_mem_batch = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
            logits = outputs["logits"]
            s_used = outputs["s_used"]
            q_heads = outputs["q_heads"]
            all_s_used.append(torch.stack(s_used).mean().item())

            v_seq_len = logits.shape[1]
            v_batch_size = logits.shape[0]
            v_flops_batch= compute_flops(s_used, v_seq_len, v_batch_size, D_MODEL, NUM_HEADS, training=training)
            v_carbon_batch = compute_carbon(flops)
            v_activation_memory = compute_activation_memory(input_ids, s_used, D_MODEL, NUM_HEADS, training=training)

            val_flops += v_flops_batch
            val_carbon += v_carbon_batch
            val_mem += v_activation_memory
            val_peak_mem_sum += v_peak_mem_batch

            loss_task = compute_loss(logits, labels)
            loss_budget = budget_loss(s_used)
            loss_entropy = beta * sum((q * torch.log(q + 1e-12)).sum(dim=-1).mean() for q in q_heads)
            loss = loss_task + loss_budget + loss_entropy

            total_loss += loss.item()
            n_batches += 1

    avg_loss = total_loss / n_batches
    ppl = math.exp(avg_loss)

    return avg_loss, ppl, val_flops, val_carbon, val_mem, val_peak_mem_sum, n_batches, np.mean(all_s_used)

# =========================
# GENERATION (EOS STOP ✔)
# =========================
def generate(model, tokenizer, instruction, input="<noinput>", max_new_tokens=100, topk_only=False):
    model.eval()
    prompt = f"Instruction: {instruction}\nInput: {input}\nResponse:"
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)

    past_kvs = None

    all_s_used = []

    for _ in range(max_new_tokens):

        with torch.no_grad():

            outputs = model(
                input_ids=input_ids,
                past_kvs=past_kvs,
                use_cache=True,
                topk_only=topk_only
            )

        logits = outputs["logits"][:, -1, :]
        past_kvs = outputs["past_key_values"]

        # =========================
        # s_used tracking
        # =========================
        s_val = outputs["s_used"]
        all_s_used.append(torch.stack(s_val).mean().item())

        probs = torch.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, 1)

        input_ids = torch.cat([input_ids, next_token], dim=1)

        if next_token.item() == tokenizer.eos_token_id:
            break

    text = tokenizer.decode(input_ids[0], skip_special_tokens=True)

    avg_s_used = np.mean(all_s_used)

    return text, avg_s_used

rouge = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
def evaluate_generation(model, tokenizer, dataset, eval_gen_samples=300):

    model.eval()

    bleu_scores = []
    rouge_scores = []
    preds = []
    refs = []

    samples = dataset["validation"].select(range(eval_gen_samples))

    for ex in samples:

        instruction = ex["instruction"]
        inp = ex["input"]
        ref = ex["output"]

        pred, _ = generate(model, tokenizer, instruction, inp, topk_only=True)

        preds.append(pred)
        refs.append(ref)

        bleu_scores.append(sentence_bleu([ref.split()], pred.split()))
        rouge_scores.append(rouge.score(ref, pred)["rougeL"].fmeasure)

    P, R, F1 = bertscore(preds, refs, lang="en", verbose=False)
    return {
        "BLEU": sum(bleu_scores)/len(bleu_scores),
        "ROUGE-L": sum(rouge_scores)/len(rouge_scores),
        "BERTScore": F1.mean().item()
    }

# =========================
# MAIN
# =========================
if __name__ == "__main__":

    # TOKENIZER
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    eos_id = tokenizer.eos_token_id

    # DATASET
    dataset = load_dataset("lavita/AlpaCare-MedInstruct-52k")
    split = dataset["train"].train_test_split(
        test_size=5000,  # 5%
        seed=42
    )
    dataset = DatasetDict({
        "train": split["train"],
        "validation": split["test"]
    })

    dataset = dataset.map(format_example)
    dataset = dataset.filter(
        lambda x: x["instruction"] is not None and len(x["instruction"]) > 0 and x["output"] is not None and len(
            x["output"]) > 0)

    tokenized = dataset.map(
        lambda x: tokenize(x, tokenizer, eos_id),
        batched=False,
        remove_columns=dataset["train"].column_names
    )

    # DATALOADER
    train_loader = DataLoader(
        tokenized["train"],
        batch_size=batch_size,
        shuffle=True,
        collate_fn=lambda batch: collate_fn(batch, tokenizer.eos_token_id)
    )

    val_loader = DataLoader(
        tokenized["validation"],
        batch_size=batch_size,
        collate_fn=lambda batch: collate_fn(batch, tokenizer.eos_token_id)
    )

    # MODEL
    hf_model = GPT2LMHeadModel.from_pretrained("gpt2")
    model = MyGPT2LM(hf_model.config).to(device)
    missing, unexpected = model.load_state_dict(
        hf_model.state_dict(),
        strict=False
    )
    print("Missing keys:", missing)
    print("Unexpected keys:", unexpected)

    optimizer = AdamW(model.parameters(), lr=lr)

    best_val_loss = float("inf")
    global_step = 0

    steps_per_epoch = len(train_loader)
    total_steps = epochs * steps_per_epoch
    log_every = max(1, total_steps // 100)

    print("Steps per epoch:", steps_per_epoch)
    print("Total steps:", total_steps)
    print("Log every:", log_every)

    weight_memory = compute_weight_memory(model)
    print(f"Model weight memory: {weight_memory:.2f} MB")

    total_carbon = 0.0
    total_steps = epochs * len(train_loader)
    step = 0

    for epoch in range(epochs):
        train_flops_sum, train_carbon_sum, train_mem_sum = 0, 0, 0
        train_peak_mem_sum, train_batches = 0, 0
        model.train()
        optimizer.zero_grad()
        all_s_used = []
        for batch in train_loader:

            step += 1
            tau = tau_schedule(step, total_steps)
            beta = beta_schedule(step, total_steps)
            sigma = sigma_schedule(step, total_steps)

            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            torch.cuda.reset_peak_memory_stats(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, tau=tau, sigma=sigma)
            peak_mem = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
            logits = outputs["logits"]
            q_heads = outputs["q_heads"]
            s_used = outputs["s_used"]

            loss_task = compute_loss(logits, labels)
            loss_budget = budget_loss(s_used)
            loss_entropy = beta * sum((q * torch.log(q + 1e-12)).sum(dim=-1).mean() for q in q_heads)
            loss = loss_task + loss_budget + loss_entropy

            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()
            optimizer.zero_grad()

            seq_len = logits.shape[1]
            batch_size = logits.shape[0]
            flops = compute_flops(s_used, seq_len, batch_size, D_MODEL, NUM_HEADS, training=True)
            carbon = compute_carbon(flops)
            mem = compute_activation_memory(input_ids, s_used, D_MODEL, NUM_HEADS, training=True)

            train_flops_sum += flops
            train_carbon_sum += carbon
            train_mem_sum += mem
            train_peak_mem_sum += peak_mem
            total_carbon += carbon

            global_step += 1
            all_s_used.append(torch.stack(s_used).mean().item())

            # =========================
            # LOG + EVAL
            # =========================
            if global_step % log_every == 0:
                print(f"\n[STEP {global_step}]")
                print(f"Train Loss: {loss.item():.4f} | Train s_mean {np.mean(all_s_used)}")
                print(f"TRAIN batches: {train_batches}")
                print(f"TRAIN Total FLOPs: {train_flops_sum:.2e}")
                print(f"TRAIN Total Carbon (gCO2): {train_carbon_sum:.6f}")
                print(f"TRAIN Avg Activation memory (MB): {train_mem_sum / train_batches:.2f}")
                print(f"TRAIN Peak GPU memory (MB): {train_peak_mem_sum / train_batches:.2f}")
                print(f"TRAIN Weight memory (MB): {weight_memory:.2f}")
                val_loss, val_ppl, val_flops, val_carbon, val_mem, val_peak_mem_sum, n_batches, s_val = evaluate_ppl(model, val_loader, device, tau=tau)
                print(f"VAL batches: {n_batches} | Val Loss  : {val_loss:.4f} | Val PPL   : {val_ppl:.2f} | Val s_mean {s_val}")
                print(f"VAL Total FLOPs: {val_flops:.2e}")
                print(f"VAL Total Carbon (gCO2): {val_carbon:.6f}")
                print(f"VAL Avg Activation memory (MB): {val_mem / n_batches:.2f}")
                print(f"VAL Peak GPU memory (MB): {val_peak_mem_sum / n_batches:.2f}")
                bleu, rouge, bert_score = evaluate_generation(model, tokenizer, dataset)
                print(f"Bleu: {bleu:.4f} | Rouge L: {rouge:.4f} | BER score: {bert_score:.4f}")

                train_flops_sum, train_carbon_sum, train_mem_sum = 0, 0, 0
                train_peak_mem_sum, train_batches = 0, 0
                all_s_used = []

                # save best model
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    torch.save(model.state_dict(), "model.pt")
                    print("✅ Best model saved")

        print(f"Epoch {epoch + 1} done")


    # =========================
    # FINAL EVALUATION
    # =========================
    print("\n🔍 FINAL EVALUATION")

    val_loss, val_ppl, val_flops, val_carbon, val_mem, val_peak_mem_sum, n_batches, s_val = evaluate_ppl(model, val_loader, device, top_k=True, training=False)

    print(f"===== FINAL RESULTS  (S_mean = {s_val})=====")
    print(f"Total carbon Train {total_carbon:.4f} gCO2")
    print(f"VAL batches: {n_batches} | Val Loss  : {val_loss:.4f} | Val PPL   : {val_ppl:.2f}")
    print(f"VAL Total FLOPs: {val_flops:.2e}")
    print(f"VAL Total Carbon (gCO2): {val_carbon:.6f}")
    print(f"VAL Avg Activation memory (MB): {val_mem / n_batches:.2f}")
    print(f"VAL Peak GPU memory (MB): {val_peak_mem_sum / n_batches:.2f}")
    bleu, rouge, bert_score = evaluate_generation(model, tokenizer, dataset)
    print(f"Bleu: {bleu:.4f} | Rouge L: {rouge:.4f} | BER score: {bert_score:.4f}")

    torch.save(model.state_dict(), "model_final.pt")
    print("💾 Final model saved")

    print("\n=========================")
    print("🧠 FINAL GENERATIONS (WIKITEXT STYLE)")
    print("=========================\n")

    prompts = [
        (
            "Provide a diagnosis based on the following clinical description",
            "A 65-year-old male presents with chest pain radiating to the left arm, sweating, and shortness of breath."
        ),
        (
            "Summarize the following medical case",
            "A 45-year-old woman with a history of hypertension presents with severe headache and blurred vision. Blood pressure is 190/120 mmHg."
        ),
        (
            "Suggest a treatment plan for the following condition",
            "Patient diagnosed with type 2 diabetes mellitus with HbA1c of 9.2% despite metformin therapy."
        ),
        (
            "Explain the most likely diagnosis",
            "A child presents with fever, barking cough, and inspiratory stridor at night."
        ),
        (
            "Interpret the following laboratory results",
            "Hemoglobin: 8 g/dL, MCV: 70 fL, Ferritin: low"
        ),
        (
            "What is the differential diagnosis for the following symptom?",
            "Chronic cough lasting more than 3 months in a non-smoker."
        ),
        (
            "Explain the mechanism of action of the drug mentioned",
            "Metformin used in type 2 diabetes treatment."
        ),
        (
            "Provide clinical advice based on the case",
            "Patient presents with fever, rash, and joint pain after travel to tropical region."
        ),
        (
            "Summarize the imaging findings",
            "CT scan shows a mass in the left lung with mediastinal lymphadenopathy."
        ),
        (
            "What is the most likely cause?",
            "Acute onset of dyspnea, pleuritic chest pain, and tachycardia after long flight."
        ),
    ]

    for i, (instruction, input) in enumerate(prompts):
        output, s_used = generate(
            model=model,
            tokenizer=tokenizer,
            instruction=instruction,
            input=input,
            max_new_tokens=100,
            topk_only=True
        )

        print(f"\n--- SAMPLE: {i + 1}  s_mean : {s_used} ---\n ---")
        print(f"INSTRUCTION: {instruction}")
        print(f"INPUT: {input}")
        print(f"OUTPUT:\n{output}")