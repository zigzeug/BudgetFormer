import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from datasets import load_dataset
from transformers import  AutoTokenizer, LongformerModel
from transformers.models.longformer.modeling_longformer import (
    LongformerIntermediate,
    LongformerOutput,
    LongformerSelfOutput,
    LongformerEmbeddings,
    LongformerPooler
)
from torch.utils.data import DataLoader
import numpy as np
from torch.optim import AdamW

# =========================
# Hyperparams
# =========================
BATCH_SIZE = 16
LR = 2e-5
NUM_EPOCHS = 10
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

NUM_HEADS = 8
D_MODEL = 768
NUM_BLOCKS = 4

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
            return out, q_head, s_used

# =========================================================
# ProfileFormer
# =========================================================
class MyLongformerAttention(nn.Module):
    def __init__(self, config):
        super().__init__()

        # 🔥 Ton attention
        self.self = AdaptiveAttention(
            d_model=config.hidden_size,
            num_heads=config.num_attention_heads
        )

        # 🔥 IMPORTANT → garder output HF
        self.output = LongformerSelfOutput(config)

    def forward(
        self,
        hidden_states,
        attention_mask=None,

        tau=1.0,
        sigma=0.0,
        s_override=None,
        topk_only=False,
        use_cache=False,
        global_attention_mask=None,
    ):
        attn_output, q_head, s_used = self.self(
            hidden_states,
            attention_mask=attention_mask,
            causal=False,
            tau=tau,
            sigma=sigma,
            s_override=s_override,
            topk_only=topk_only,
            use_cache=use_cache
        )

        attention_output = self.output(attn_output, hidden_states)

        return attention_output, q_head, s_used

class MyLongformerLayer(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.attention = MyLongformerAttention(config)
        self.intermediate = LongformerIntermediate(config)
        self.output = LongformerOutput(config)

    def forward(
        self,
        hidden_states,
        attention_mask=None,

        tau=1.0,
        sigma=0.0,
        s_override=None,
        topk_only=False,
        use_cache=False,

        global_attention_mask=None,
    ):
        attn_output, q_head, s_used = self.attention(
            hidden_states,
            attention_mask,
            tau,
            sigma,
            s_override,
            topk_only,
            use_cache,
            global_attention_mask
        )

        intermediate_output = self.intermediate(attn_output)
        layer_output = self.output(intermediate_output, attn_output)

        return layer_output, q_head, s_used

class MyLongformerEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.layer = nn.ModuleList([
            MyLongformerLayer(config)
            for _ in range(config.num_hidden_layers)
        ])

    def forward(
        self,
        hidden_states,
        attention_mask=None,

        tau=1.0,
        sigma=0.0,
        s_override=None,
        topk_only=False,
        use_cache=False,

        global_attention_mask=None,
    ):
        all_q, all_s = [], []

        for layer_module in self.layer:
            hidden_states, q_head, s_used = layer_module(
                hidden_states,
                attention_mask,
                tau,
                sigma,
                s_override,
                topk_only,
                use_cache,
                global_attention_mask
            )

            all_q.append(q_head)
            all_s.append(s_used)

        return hidden_states, all_q, all_s

class MyLongformerModel(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.embeddings = LongformerEmbeddings(config)
        self.encoder = MyLongformerEncoder(config)
        self.pooler = LongformerPooler(config)

    def forward(
        self,
        input_ids,
        attention_mask=None,
        token_type_ids=None,

        tau=1.0,
        sigma=0.0,
        s_override=None,
        topk_only=False,
        use_cache=False,

        global_attention_mask=None,
    ):
        embedding_output = self.embeddings(
            input_ids=input_ids,
            token_type_ids=token_type_ids
        )

        hidden_states, all_q, all_s = self.encoder(
            embedding_output,
            attention_mask,
            tau,
            sigma,
            s_override,
            topk_only,
            use_cache,
            global_attention_mask
        )

        pooled_output = self.pooler(hidden_states)

        return {
            "last_hidden_state": hidden_states,
            "pooler_output": pooled_output,
            "q_heads": all_q,
            "s_used": all_s
        }

class MyLongformerForQA(nn.Module):
    def __init__(self, config, dropout_prob=0.2):
        super().__init__()

        self.longformer = MyLongformerModel(config)

        self.qa_outputs = nn.Linear(config.hidden_size, 2)
        self.dropout = nn.Dropout(dropout_prob)

    def forward(
        self,
        input_ids,
        attention_mask=None,

        tau=1.0,
        sigma=0.0,
        s_override=None,
        topk_only=False,

        global_attention_mask=None,
    ):
        outputs = self.longformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
            tau=tau,
            sigma=sigma,
            s_override=s_override,
            topk_only=topk_only,
            global_attention_mask=global_attention_mask
        )

        sequence_output = self.dropout(outputs["last_hidden_state"])

        logits = self.qa_outputs(sequence_output)

        start_logits, end_logits = logits.split(1, dim=-1)

        return {
            "start_logits": start_logits.squeeze(-1),
            "end_logits": end_logits.squeeze(-1),
            "q_heads": outputs["q_heads"],
            "s_used": outputs["s_used"]
        }

#################################
# Dataset
#################################
dataset = load_dataset("squad_v2")
tokenizer = AutoTokenizer.from_pretrained("allenai/longformer-base-4096")

def preprocess_function(examples):
    questions = [q.strip() for q in examples["question"]]

    inputs = tokenizer(
        questions,
        examples["context"],
        max_length=384,
        truncation="only_second",
        return_offsets_mapping=True,
        padding="max_length",
    )

    offset_mapping = inputs["offset_mapping"]
    answers = examples["answers"]

    start_positions = []
    end_positions = []

    for i, offsets in enumerate(offset_mapping):

        answer = answers[i]

        # =========================
        # ❗ SQuAD v2: no answer case
        # =========================
        if len(answer["text"]) == 0:
            start_positions.append(0)
            end_positions.append(0)
            continue

        start_char = answer["answer_start"][0]
        end_char = start_char + len(answer["text"][0])

        sequence_ids = inputs.sequence_ids(i)

        # find context start/end
        idx = 0
        while sequence_ids[idx] != 1:
            idx += 1
        context_start = idx

        while idx < len(sequence_ids) and sequence_ids[idx] == 1:
            idx += 1
        context_end = idx - 1

        # =========================
        # check if answer in context
        # =========================
        if offsets[context_start][0] > end_char or offsets[context_end][1] < start_char:
            start_positions.append(0)
            end_positions.append(0)
        else:
            # start
            idx = context_start
            while idx <= context_end and offsets[idx][0] <= start_char:
                idx += 1
            start_positions.append(idx - 1)

            # end
            idx = context_end
            while idx >= context_start and offsets[idx][1] >= end_char:
                idx -= 1
            end_positions.append(idx + 1)

    inputs.pop("offset_mapping")

    inputs["start_positions"] = start_positions
    inputs["end_positions"] = end_positions

    return inputs

tokenized_dataset = dataset.map(
    preprocess_function,
    batched=True,
    remove_columns=dataset["train"].column_names
)
tokenized_dataset.set_format(
    type="torch",
    columns=["input_ids", "attention_mask", "token_type_ids", "start_positions", "end_positions"]
)
train_loader = DataLoader(
    tokenized_dataset["train"],
    batch_size=BATCH_SIZE,
    shuffle=True
)
val_loader = DataLoader(
    tokenized_dataset["validation"],
    batch_size=BATCH_SIZE
)

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
        if training == False:
          H_active = s.mean().item() * num_heads
        else:
          H_active = num_heads

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

def compute_activation_memory(ids, ss, d_model, num_heads, training=True, all_heads=True):
    B, N = ids.shape
    d_head = d_model // num_heads
    bytes_per_float = 4

    total_mem = 0

    for s in ss:
        if all_heads == False:
          H_active = s.mean().item() * num_heads
        else:
          H_active = num_heads

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

def compute_weight_memory(model):
    total_bytes = 0
    for p in model.parameters():
        total_bytes += p.numel() * p.element_size()
    return total_bytes / (1024**2)  # MB

# Exact Match (EM) → réponse parfaite
# F1 → overlap tokens
def compute_qa_metrics(start_preds, end_preds, start_true, end_true):
    em = np.mean([
        int(sp == st and ep == et)
        for sp, ep, st, et in zip(start_preds, end_preds, start_true, end_true)
    ])

    f1_scores = []
    for sp, ep, st, et in zip(start_preds, end_preds, start_true, end_true):
        pred_range = set(range(sp, ep + 1))
        true_range = set(range(st, et + 1))

        if len(pred_range) == 0 or len(true_range) == 0:
            f1_scores.append(0)
            continue

        overlap = len(pred_range & true_range)
        precision = overlap / len(pred_range)
        recall = overlap / len(true_range)

        if precision + recall == 0:
            f1_scores.append(0)
        else:
            f1_scores.append(2 * precision * recall / (precision + recall))

    return em, np.mean(f1_scores)

##################################################
# Validation
#################################################
def run_validation(model, val_loader, beta, training=True, D_MODEL=768, NUM_HEADS=8, topk_only=True, all_heads=True):
    model.eval()

    val_start_preds, val_end_preds = [], []
    val_start_true, val_end_true = [], []
    val_s = []

    val_loss = 0
    val_flops, val_carbon, val_mem = 0, 0, 0
    val_batches = 0

    with torch.no_grad():
        for vbatch in val_loader:
            val_batches += 1

            vid = vbatch["input_ids"].to(DEVICE)
            vmask = vbatch["attention_mask"].to(DEVICE)
            v_start = vbatch["start_positions"].to(DEVICE)
            v_end = vbatch["end_positions"].to(DEVICE)

            v_outputs = model(
                vid,
                vmask,
                tau=1.0,
                sigma=0.0,
                topk_only=topk_only
            )

            v_start_logits = v_outputs["start_logits"]
            v_end_logits = v_outputs["end_logits"]

            v_qs = v_outputs["q_heads"]
            v_ss = v_outputs["s_used"]

            # =========================
            # METRICS
            # =========================
            v_seq_len = vid.shape[1]
            v_batch_size = vid.shape[0]

            v_flops_batch = compute_flops(
                v_ss, v_seq_len, v_batch_size, d_model=D_MODEL, num_heads=NUM_HEADS, training=training
            )
            v_carbon_batch = compute_carbon(v_flops_batch)
            v_activation_memory = compute_activation_memory(
                vid, v_ss, d_model=D_MODEL, num_heads=NUM_HEADS, training=training, all_heads=all_heads
            )

            val_flops += v_flops_batch
            val_carbon += v_carbon_batch
            val_mem += v_activation_memory

            # =========================
            # PREDICTIONS
            # =========================
            v_sp = v_start_logits.argmax(dim=-1)
            v_ep = v_end_logits.argmax(dim=-1)

            val_start_preds.extend(v_sp.cpu().numpy())
            val_end_preds.extend(v_ep.cpu().numpy())
            val_start_true.extend(v_start.cpu().numpy())
            val_end_true.extend(v_end.cpu().numpy())

            val_s.append(torch.stack(v_ss).mean().item())

            # =========================
            # LOSS
            # =========================
            v_L = (
                F.cross_entropy(v_start_logits, v_start) +
                F.cross_entropy(v_end_logits, v_end)
            ) / 2

            v_L_budget = budget_loss(v_ss)
            v_L_entropy = beta * sum(
                (q * torch.log(q + 1e-12)).sum(dim=-1).mean()
                for q in v_qs
            )

            val_loss += (v_L + v_L_budget + v_L_entropy).item()

    # =========================
    # METRICS
    # =========================
    em, f1 = compute_qa_metrics(
        val_start_preds,
        val_end_preds,
        val_start_true,
        val_end_true
    )

    return {
        "loss": val_loss / len(val_loader),
        "em": em,
        "f1": f1,
        "s_mean": np.mean(val_s),
        "flops": val_flops,
        "carbon": val_carbon,
        "mem": val_mem / val_batches,
    }

#######################################
# Training
#######################################

if __name__ == "__main__":
    hf_model = LongformerModel.from_pretrained("allenai/longformer-base-4096")
    model = MyLongformerForQA(hf_model.config)
    missing, unexpected = model.longformer.load_state_dict(
        hf_model.state_dict(),
        strict=False
    )
    print("Missing keys:", missing)
    print("Unexpected keys:", unexpected)
    optimizer = AdamW(model.parameters(), lr=LR)
    total_steps = NUM_EPOCHS * len(train_loader)
    step = 0
    best_train_f1 = 0.0
    best_train_em = 0.0
    total_flops, total_carbon = 0, 0
    D_MODEL = hf_model.config.hidden_size
    NUM_HEADS = hf_model.config.num_attention_heads
    beta = 0.0
    weight_memory = compute_weight_memory(model)
    print(f"Model weight memory: {weight_memory:.2f} MB")

    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()

        train_start_preds, train_end_preds = [], []
        train_start_true, train_end_true = [], []
        train_s = []

        train_flops_sum, train_carbon_sum, train_mem_sum = 0, 0, 0
        train_peak_mem_sum = 0
        train_batches = 0

        for batch in train_loader:
            step += 1
            train_batches += 1

            tau = tau_schedule(step, total_steps)
            beta = beta_schedule(step, total_steps)
            sigma = sigma_schedule(step, total_steps)

            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            start_pos = batch["start_positions"].to(DEVICE)
            end_pos = batch["end_positions"].to(DEVICE)

            torch.cuda.reset_peak_memory_stats(DEVICE)
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                tau=tau,
                sigma=sigma
            )
            peak_mem_batch = torch.cuda.max_memory_allocated(DEVICE) / (1024**2)

            start_logits = outputs["start_logits"]
            end_logits = outputs["end_logits"]
            qs = outputs["q_heads"]
            ss = outputs["s_used"]

            # =========================
            # METRICS COMPUTE
            # =========================
            seq_len = input_ids.shape[1]
            batch_size = input_ids.shape[0]

            flops_batch = compute_flops(ss, seq_len, batch_size, D_MODEL, NUM_HEADS)
            carbon_batch = compute_carbon(flops_batch)
            activation_memory = compute_activation_memory(input_ids, ss, D_MODEL, NUM_HEADS)

            train_flops_sum += flops_batch
            train_carbon_sum += carbon_batch
            train_mem_sum += activation_memory
            train_peak_mem_sum += peak_mem_batch

            total_flops += flops_batch
            total_carbon += carbon_batch

            # =========================
            # PREDICTIONS
            # =========================
            start_pred = start_logits.argmax(dim=-1)
            end_pred = end_logits.argmax(dim=-1)

            train_start_preds.extend(start_pred.cpu().numpy())
            train_end_preds.extend(end_pred.cpu().numpy())
            train_start_true.extend(start_pos.cpu().numpy())
            train_end_true.extend(end_pos.cpu().numpy())
            train_s.append(torch.stack(ss).mean().item())

            # =========================
            # LOSSES
            # =========================
            L_start = F.cross_entropy(start_logits, start_pos)
            L_end = F.cross_entropy(end_logits, end_pos)
            L_task = (L_start + L_end) / 2

            L_budget = budget_loss(ss)
            L_entropy = beta * sum((q * torch.log(q + 1e-12)).sum(dim=-1).mean() for q in qs)

            loss = L_task + L_budget + L_entropy

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # =========================
            # PRINT
            # =========================
            is_last_step = (epoch == NUM_EPOCHS and step == total_steps)
            if (step % 300 == 0) or is_last_step:

                em, f1 = compute_qa_metrics(
                    train_start_preds,
                    train_end_preds,
                    train_start_true,
                    train_end_true
                )

                print(f"\nStep {step}")
                print(f"Loss {loss.item():.4f} | L_task {L_task.item():.4f} | "
                      f"L_budget {L_budget.item():.4f} | L_entropy {L_entropy:.4f}")

                print(f"TRAIN → EM {em:.3f} | F1 {f1:.3f} | s_mean {np.mean(train_s):.3f}")

                print(f"TRAIN FLOPs: {train_flops_sum:.2e}")
                print(f"TRAIN Carbon: {train_carbon_sum:.6f}")
                print(f"TRAIN Mem: {train_mem_sum / train_batches:.2f} MB")
                print(f"TRAIN Peak Mem: {train_peak_mem_sum/train_batches:.2f} MB")

                # reset
                train_start_preds, train_end_preds = [], []
                train_start_true, train_end_true = [], []
                train_s = []

                train_flops_sum, train_carbon_sum, train_mem_sum = 0, 0, 0
                train_peak_mem_sum = 0
                train_batches = 0

                # =========================
                # SAVE
                # =========================
                if f1 > best_train_f1 and em > best_train_em:
                    best_val_f1 = f1
                    best_val_em = em
                    torch.save(model.state_dict(), "best_qa_model.pt")
                    print("NEW BEST MODEL SAVED")

                if f1 > best_train_f1:
                    best_train_f1 = f1

                if em > best_train_em:
                    best_train_em = em

                model.train()

    torch.save(model.state_dict(), "final_qa_model.pt")

    print(f"Training finished. Best F1: {best_train_f1:.4f}. Best EM: {best_train_em:4f}.  TOTAL TRAINING CARBON: {total_carbon:.4f} gCO2")

    print("\n==============================")
    print("STARTING TEST PHASE TOPK ONLY = FALSE")
    print("==============================\n")
    val_metrics = run_validation(model, val_loader, beta, training=False, D_MODEL=D_MODEL, NUM_HEADS=NUM_HEADS, topk_only=False, all_heads=True)
    print(
        f"VAL Loss {val_metrics['loss']:.4f} | EM {val_metrics['em']:.3f} | F1 {val_metrics['f1']:.3f} | s_mean {val_metrics['s_mean']:.3f}")
    print(f"VAL Total FLOPs: {val_metrics['flops']:.2e}")
    print(f"VAL Carbon: {val_metrics['carbon']:.6f} gCO2")
    print(f"VAL Mem: {val_metrics['mem']:.2f} MB")

    print("\n==============================")
    print("STARTING TEST PHASE TOPK ONLY = TRUE")
    print("==============================\n")
    val_metrics = run_validation(model, val_loader, beta, training=False, D_MODEL=D_MODEL, NUM_HEADS=NUM_HEADS, topk_only=True, all_heads=False)
    print(
        f"VAL Loss {val_metrics['loss']:.4f} | EM {val_metrics['em']:.3f} | F1 {val_metrics['f1']:.3f} | s_mean {val_metrics['s_mean']:.3f}")
    print(f"VAL Total FLOPs: {val_metrics['flops']:.2e}")
    print(f"VAL Carbon: {val_metrics['carbon']:.6f} gCO2")
    print(f"VAL Mem: {val_metrics['mem']:.2f} MB")