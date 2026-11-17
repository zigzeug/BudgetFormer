import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer
import math
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
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


def compute_metrics(all_preds, all_true, all_probs):
    acc = accuracy_score(all_true, all_preds)
    f1 = f1_score(all_true, all_preds, average="macro", zero_division=0)
    prec = precision_score(all_true, all_preds, average="macro", zero_division=0)
    rec = recall_score(all_true, all_preds, average="macro", zero_division=0)
    true_onehot = F.one_hot(torch.tensor(all_true), num_classes=all_probs.shape[1]).numpy()
    try:
        auc = roc_auc_score(true_onehot, all_probs, multi_class="ovo", average="macro")
    except ValueError:
        auc = float("nan")
    return acc, f1, prec, rec, auc

# =========================
# Adaptive Attention
# =========================

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


# =========================
# Adaptive Block
# =========================
class AdaptiveBlock(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.attn = AdaptiveAttention(d_model, num_heads)
        self.norm1 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.ReLU(),
            nn.Linear(4 * d_model, d_model)
        )
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self,
                X,
                attention_mask = None,
                causal = False,
                tau = 1.0,
                sigma = 0.0,
                s_override = None,
                topk_only = False,
                past_k = None,
                past_v = None,
                use_cache = False
                ):
        attn_out, q, s_used = self.attn(X, attention_mask, causal, tau, sigma, s_override, topk_only, past_k, past_v,
                use_cache)

        X = self.norm1(X + attn_out)
        ff_out = self.ff(X)
        X = self.norm2(X + ff_out)
        return X, q, s_used


# =========================
# Positional Encoding
# =========================
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.pe = pe.unsqueeze(0)  # (1, max_len, d_model)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)].to(x.device)


# =========================
# Adaptive Classifier (FIXED)
# =========================
class AdaptiveClassifier(nn.Module):
    def __init__(self, vocab_size, num_classes):
        super().__init__()

        # 🔥 Embedding au lieu de BERT
        self.embedding = nn.Embedding(vocab_size, D_MODEL)

        # 🔥 Positional encoding
        self.pos_encoding = PositionalEncoding(D_MODEL)

        # 🔥 Tes blocs transformer adaptatifs
        self.blocks = nn.ModuleList([
            AdaptiveBlock(D_MODEL, NUM_HEADS) for _ in range(NUM_BLOCKS)
        ])

        self.classifier = nn.Linear(D_MODEL, num_classes)

    def forward(self,
                input_ids,
                attention_mask = None,
                causal = False,
                tau = 1.0,
                sigma = 0.0,
                s_override = None,
                topk_only = False,
                past_k = None,
                past_v = None,
                use_cache = False
                ):

        # Embedding
        X = self.embedding(input_ids)  # (B, N, D)

        # Positional encoding
        X = self.pos_encoding(X)

        qs, ss = [], []

        # Pass through adaptive transformer blocks
        for block in self.blocks:
            X, q, s_used = block(X, attention_mask, causal, tau, sigma, s_override, topk_only, past_k, past_v, use_cache)
            qs.append(q)
            ss.append(s_used)

        # Pooling
        pooled = X.mean(dim=1)

        logits = self.classifier(pooled)

        return logits, qs, ss


def load_data(dataset_name="snli", batch_size=32, max_length=128):
    print(f"Chargement du dataset {dataset_name}...")
    dataset = load_dataset(dataset_name)

    print("Chargement du tokenizer bert-base-uncased...")
    tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-uncased")

    # Shuffle AVANT le select pour avoir un échantillon représentatif
    dataset = dataset.shuffle(seed=42)

    # =========================
    # Subset AVANT tokenization
    # =========================
    if dataset_name == "snli":
        dataset = dataset.filter(lambda example: example['label'] != -1)

        #train_dataset = dataset["train"].select(range(100))
        #val_dataset = dataset["validation"].select(range(50))
        #test_dataset = dataset["test"].select(range(50))

        train_dataset = dataset["train"]
        val_dataset = dataset["validation"]
        test_dataset = dataset["test"]

        num_classes = 3

        def tokenize_function(examples):
            return tokenizer(
                examples["premise"],
                examples["hypothesis"],
                padding="max_length",
                truncation=True,
                max_length=max_length
            )
    else:
        if dataset_name == "imdb":
            text_col, num_classes = "text", 2
        elif dataset_name == "ag_news":
            text_col, num_classes = "text", 4
        elif dataset_name == "yelp_review_full":
            text_col, num_classes = "text", 5
        elif dataset_name == "dbpedia_14":
            text_col, num_classes = "content", 14
        else:
            raise ValueError("Dataset non supporté")

        #train_dataset = dataset["train"].select(range(100))
        train_dataset = dataset["train"]

        if "validation" in dataset:
            #val_dataset = dataset["validation"].select(range(50))
            val_dataset = dataset["validation"]
        else:
            #val_dataset = dataset["test"].select(range(50))
            val_dataset = dataset["test"]

        if "test" in dataset:
            #test_dataset = dataset["test"].select(range(50))
            test_dataset = dataset["test"]

        else:
            test_dataset = val_dataset

        def tokenize_function(examples):
            return tokenizer(
                examples[text_col],
                padding="max_length",
                truncation=True,
                max_length=max_length
            )

    # =========================
    # Tokenisation seulement sur les subsets
    # =========================
    print("Tokenisation en cours...")
    train_dataset = train_dataset.map(tokenize_function, batched=True)
    val_dataset = val_dataset.map(tokenize_function, batched=True)
    test_dataset = test_dataset.map(tokenize_function, batched=True)

    # Format PyTorch
    train_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
    val_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
    test_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])

    # DataLoaders
    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size)

    return train_dataloader, val_dataloader, test_dataloader, tokenizer.vocab_size, num_classes


# =========================
# Training
# =========================
#train_loader, val_loader, test_loader, vocab_size, num_classes = load_data("dbpedia_14", BATCH_SIZE, 64)
#train_loader, val_loader, test_loader, vocab_size, num_classes = load_data("ag_news", BATCH_SIZE, 64)
#train_loader, val_loader, test_loader, vocab_size, num_classes = load_data("snli", BATCH_SIZE, 64)
#train_loader, val_loader, test_loader, vocab_size, num_classes = load_data("imdb", BATCH_SIZE, 512)
train_loader, val_loader, test_loader, vocab_size, num_classes = load_data("yelp_review_full", BATCH_SIZE, 512)





model = AdaptiveClassifier(vocab_size, num_classes).to(DEVICE)
optimizer = AdamW(model.parameters(), lr=LR)
total_steps = NUM_EPOCHS * len(train_loader)
step = 0
best_val_acc = 0.0

# 🔥 metrics globaux
total_flops, total_carbon = 0, 0

# mémoire des poids (fixe)
weight_memory = compute_weight_memory(model)
print(f"Model weight memory: {weight_memory:.2f} MB")

for epoch in range(1, NUM_EPOCHS+1):
    model.train()

    train_preds, train_true, train_probs, train_s = [], [], [], []

    # accumulateurs train metrics
    train_flops_sum, train_carbon_sum, train_mem_sum = 0, 0, 0
    train_batches = 0
    train_peak_mem_sum = 0

    for batch in train_loader:
        step += 1
        train_batches += 1

        tau = tau_schedule(step, total_steps)
        beta = beta_schedule(step, total_steps)
        sigma = sigma_schedule(step, total_steps)

        input_ids = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)
        labels = batch["label"].to(DEVICE)

        # reset peak memory
        torch.cuda.reset_peak_memory_stats(DEVICE)

        logits, qs, ss = model(
                             input_ids=input_ids,
                             attention_mask=attention_mask,
                             causal=False,
                             tau=tau,
                             sigma=sigma,
                             s_override=None,
                             topk_only=False,
                             past_k=None,
                             past_v=None,
                             use_cache=False
                         )

        # peak mémoire GPU pour ce batch
        peak_mem_batch = torch.cuda.max_memory_allocated(DEVICE) / (1024**2)  # MB

        # =========================
        # METRICS TRAIN
        # =========================
        seq_len = input_ids.shape[1]
        batch_size = input_ids.shape[0]

        flops_batch = compute_flops(ss, seq_len, batch_size, D_MODEL, NUM_HEADS)
        carbon_batch = compute_carbon(flops_batch)
        activation_memory = compute_activation_memory(input_ids, ss, D_MODEL, NUM_HEADS)


        train_peak_mem_sum += peak_mem_batch

        train_flops_sum += flops_batch
        train_carbon_sum += carbon_batch
        train_mem_sum += activation_memory

        total_flops += flops_batch
        total_carbon += carbon_batch

        probs = F.softmax(logits, dim=-1)
        preds = logits.argmax(dim=-1)

        train_preds.extend(preds.detach().cpu().numpy())
        train_true.extend(labels.detach().cpu().numpy())
        train_probs.extend(probs.detach().cpu().numpy())
        train_s.append(torch.stack(ss).mean().item())

        # losses
        L_task = F.cross_entropy(logits, labels)
        L_budget = budget_loss(ss)
        L_entropy = beta * sum((q * torch.log(q+1e-12)).sum(dim=-1).mean() for q in qs)
        loss = L_task + L_budget + L_entropy

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # =========================
        # PRINT
        # =========================
        is_last_step = (epoch == NUM_EPOCHS and step == total_steps)

        if (step % 300 == 0) or is_last_step:

            train_probs_np = np.array(train_probs)

            acc, f1, prec, rec, auc = compute_metrics(
                train_preds, train_true, train_probs_np
            )

            print(f"\nStep {step}")
            print(f"Loss {loss.item():.4f} | L_task {L_task.item():.4f} | "
                  f"L_budget {L_budget.item():.4f} | L_entropy {L_entropy:.4f}")

            print(f"TRAIN → Acc {acc:.3f} | F1 {f1:.3f} | Prec {prec:.3f} | "
                  f"Recall {rec:.3f} | AUC {auc:.3f} | s_mean {np.mean(train_s):.3f}")

            # TRAIN metrics agrégées
            print(f"TRAIN batches: {train_batches}")
            print(f"TRAIN Total FLOPs: {train_flops_sum:.2e}")
            print(f"TRAIN Total Carbon (gCO2): {train_carbon_sum:.6f}")
            print(f"TRAIN Avg Activation memory (MB): {train_mem_sum/train_batches:.2f}")
            print(f"TRAIN Peak GPU memory (MB): {train_peak_mem_sum/train_batches:.2f}")
            print(f"TRAIN Weight memory (MB): {weight_memory:.2f}")

            # reset accumulateurs train
            train_preds, train_true, train_probs, train_s = [], [], [], []
            train_flops_sum, train_carbon_sum, train_mem_sum, train_peak_mem_sum = 0, 0, 0, 0
            train_batches = 0

            # =========================
            # VALIDATION
            # =========================
            model.eval()

            val_loss = 0
            val_preds, val_true, val_probs, val_s = [], [], [], []

            # accumulateurs validation
            val_flops, val_carbon, val_mem, val_peak_mem_sum = 0, 0, 0, 0
            val_batches = 0

            with torch.no_grad():
                for vbatch in val_loader:
                    val_batches += 1

                    vid = vbatch["input_ids"].to(DEVICE)
                    vmask = vbatch["attention_mask"].to(DEVICE)
                    vlabels = vbatch["label"].to(DEVICE)

                    # reset peak mémoire
                    torch.cuda.reset_peak_memory_stats(DEVICE)

                    v_logits, v_qs, v_ss = model(vid, vmask, tau=tau, sigma=0.0)

                    # peak mémoire GPU validation batch
                    v_peak_mem_batch = torch.cuda.max_memory_allocated(DEVICE) / (1024**2)  # MB


                    # METRICS VALIDATION
                    v_seq_len = vid.shape[1]
                    v_batch_size = vid.shape[0]

                    v_flops_batch = compute_flops(v_ss, v_seq_len, v_batch_size, D_MODEL, NUM_HEADS)
                    v_carbon_batch = compute_carbon(v_flops_batch)
                    v_activation_memory = compute_activation_memory(vid, v_ss, D_MODEL, NUM_HEADS)


                    val_flops += v_flops_batch
                    val_carbon += v_carbon_batch
                    val_mem += v_activation_memory
                    val_peak_mem_sum += v_peak_mem_batch

                    v_probs = F.softmax(v_logits, dim=-1)
                    v_preds = v_logits.argmax(dim=-1)

                    val_preds.extend(v_preds.cpu().numpy())
                    val_true.extend(vlabels.cpu().numpy())
                    val_probs.extend(v_probs.cpu().numpy())
                    val_s.append(torch.stack(v_ss).mean().item())

                    v_L_task = F.cross_entropy(v_logits, vlabels)
                    v_L_budget = budget_loss(v_ss)
                    v_L_entropy = beta * sum((q*torch.log(q+1e-12)).sum(dim=-1).mean() for q in v_qs)

                    val_loss += (v_L_task + v_L_budget + v_L_entropy).item()

            val_probs_np = np.array(val_probs)

            val_acc, val_f1, val_prec, val_rec, val_auc = compute_metrics(
                val_preds, val_true, val_probs_np
            )

            print(f"\nVAL → Loss {val_loss/len(val_loader):.4f} | "
                  f"Acc {val_acc:.3f} | F1 {val_f1:.3f} | "
                  f"Prec {val_prec:.3f} | Recall {val_rec:.3f} | "
                  f"AUC {val_auc:.3f} | s_mean {np.mean(val_s):.3f}")

            # VALIDATION metrics agrégées
            print(f"VAL batches: {val_batches}")
            print(f"VAL Total FLOPs: {val_flops:.2e}")
            print(f"VAL Total Carbon (gCO2): {val_carbon:.6f}")
            print(f"VAL Avg Activation memory (MB): {val_mem/val_batches:.2f}")
            print(f"VAL Peak GPU memory (MB): {val_peak_mem_sum/val_batches:.2f}")

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(model.state_dict(), "best_adaptive_model.pt")
                print("NEW BEST MODEL SAVED")

            model.train()

torch.save(model.state_dict(), "final_adaptive_model.pt")
print(f"Training finished. Best val acc: {best_val_acc:.4f}. Total carbon Train {total_carbon:.4f} gCO2")

# =========================
# Test / Inference
# =========================
print("\n==============================")
print("STARTING TEST PHASE")
print("==============================\n")

def evaluate(loader, tau, sigma, s_override=None, topk_only=False):
    all_preds, all_true, all_probs, all_s = [], [], [], []
    qs_list, ss_list = [], []

    # accumulateurs globaux
    total_flops, total_carbon, total_activation_mem, total_peak_mem = 0, 0, 0, 0
    total_batches = 0

    with torch.no_grad():
        for batch in loader:
            total_batches += 1

            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels = batch["label"].to(DEVICE)

            # reset peak GPU memory pour ce batch
            torch.cuda.reset_peak_memory_stats(DEVICE)

            logits, qs, ss = model(
                             input_ids=input_ids,
                             attention_mask=attention_mask,
                             causal=False,
                             tau=tau,
                             sigma=sigma,
                             s_override=s_override,
                             topk_only=topk_only,
                             past_k=None,
                             past_v=None,
                             use_cache=False
                         )

            # peak mémoire GPU réelle batch
            peak_mem_batch = torch.cuda.max_memory_allocated(DEVICE) / (1024**2)  # MB


            # =========================
            # METRICS
            # =========================
            seq_len = input_ids.shape[1]
            batch_size = input_ids.shape[0]

            flops_batch = compute_flops(ss, seq_len, batch_size, D_MODEL, NUM_HEADS, training=False)
            carbon_batch = compute_carbon(flops_batch)
            activation_memory = compute_activation_memory(input_ids, ss, D_MODEL, NUM_HEADS, training=False)


            total_flops += flops_batch
            total_carbon += carbon_batch
            total_activation_mem += activation_memory
            total_peak_mem += peak_mem_batch

            probs = F.softmax(logits, dim=-1)
            preds = logits.argmax(dim=-1)

            all_preds.extend(preds.cpu().numpy())
            all_true.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            all_s.append(torch.stack(ss).mean().item())

            qs_list = qs
            ss_list = ss

    all_probs = np.array(all_probs)

    acc, f1, prec, rec, auc = compute_metrics(all_preds, all_true, all_probs)
    s_mean = sum(all_s) / len(all_s)

    avg_activation_mem = total_activation_mem / total_batches
    avg_peak_mem = total_peak_mem / total_batches

    return acc, f1, prec, rec, auc, s_mean, qs_list, ss_list, total_flops, total_carbon, avg_activation_mem, avg_peak_mem, total_batches

# =========================
# AUTOMATIC BUDGET
# =========================
print("\n--- AUTOMATIC BUDGET ---")

for topk in [False, True]:
    print(f"\nMode topk_only={topk}")

    acc, f1, prec, rec, auc, s_mean, qs, ss, flops, carbon, act_mem, peak_mem, nb_batches = evaluate(
        test_loader, tau=TAU_MIN, sigma=0.0, topk_only=topk
    )

    print(f"Accuracy: {acc:.4f} | F1: {f1:.4f} | Precision: {prec:.4f} | "
          f"Recall: {rec:.4f} | AUC: {auc:.4f} | s_mean: {s_mean:.4f}")

    print(f"FLOPs total: {flops:.2e} | over {nb_batches} batches")
    print(f"Carbon total (gCO2): {carbon:.6f}")
    print(f"Avg activation memory (MB): {act_mem:.2f}")
    print(f"Avg peak GPU memory per batch (MB): {peak_mem:.2f}")

    print("Example head probs (first sample):")
    for i, q_block in enumerate(qs):
        print(f"Block {i} head probs:", q_block[0].cpu().tolist())
        if topk:
            k = max(int(ss[i][0].item()*NUM_HEADS),1)
            topk_idx = torch.topk(q_block, k, dim=-1)[1]
            print(f"Block {i} top-{k} heads:", topk_idx[0].cpu().tolist())


# =========================
# USER CONTROLLED BUDGET
# =========================
s_override_list = [0.1, 0.25, 0.5, 0.75, 1.0]

print("\n--- USER CONTROLLED BUDGET ---")

for s_val in s_override_list:
    print(f"\ns_override={s_val}")

    for topk in [False, True]:
        print(f"\nMode topk_only={topk}")

        acc, f1, prec, rec, auc, s_mean, qs, ss, flops, carbon, act_mem, peak_mem, nb_batches = evaluate(
            test_loader,
            tau=TAU_MIN,
            sigma=0.0,
            s_override=torch.tensor(s_val).to(DEVICE),
            topk_only=topk
        )

        print(f"Accuracy: {acc:.4f} | F1: {f1:.4f} | Precision: {prec:.4f} | "
              f"Recall: {rec:.4f} | AUC: {auc:.4f} | s_mean: {s_mean:.4f}")

        print(f"FLOPs total: {flops:.2e} | over {nb_batches} batches")
        print(f"Carbon total (gCO2): {carbon:.6f}")
        print(f"Avg activation memory (MB): {act_mem:.2f}")
        print(f"Avg peak GPU memory per batch (MB): {peak_mem:.2f}")

        print("Example head probs (first sample):")
        for i, q_block in enumerate(qs):
            print(f"Block {i} head probs:", q_block[0].cpu().tolist())
            if topk:
                k = max(int(s_val*NUM_HEADS),1)
                topk_idx = torch.topk(q_block, k, dim=-1)[1]
                print(f"Block {i} top-{k} heads:", topk_idx[0].cpu().tolist())

print("\n==============================\nTEST FINISHED\n==============================")


