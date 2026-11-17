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
from collections import defaultdict

# =========================
# Hyperparams
# =========================
BATCH_SIZE = 16
LR = 2e-5
NUM_EPOCHS = 10
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

NUM_HEADS = 12
D_MODEL = 768
NUM_BLOCKS = 4

# =========================
# METRICS
# =========================
def compute_flops(seq_len, batch_size, d_model, num_heads, num_blocks):
    d_head = d_model // num_heads
    H_active = num_heads  # baseline : toutes les têtes actives

    # 1. QKV projections
    flops_qkv = 3 * batch_size * seq_len * d_model * d_model

    # 2. QK^T
    flops_qk = batch_size * H_active * (seq_len ** 2) * d_head

    # 3. Softmax
    flops_softmax = batch_size * H_active * (seq_len ** 2)

    # 4. AV
    flops_av = batch_size * H_active * (seq_len ** 2) * d_head

    # 5. Output projection
    flops_out = batch_size * seq_len * d_model * d_model

    # Total par bloc
    flops_per_block = flops_qkv + flops_qk + flops_softmax + flops_av + flops_out

    # Total pour tous les blocs
    return flops_per_block * num_blocks


def compute_activation_memory(ids, d_model, num_heads, num_blocks, training=True):
    """
    ids: (B, N)
    """
    B, N = ids.shape
    d_head = d_model // num_heads
    H_active = num_heads
    bytes_per_float = 4

    # 1. Input
    mem_input = B * N * d_model

    # 2. QKV
    mem_qkv = 3 * B * N * d_model

    # 3. Attention scores
    mem_scores = B * H_active * (N * N)

    # 4. Softmax
    mem_attn = B * H_active * (N * N)

    # 5. Head outputs
    mem_heads = B * N * H_active * d_head

    # 6. Output
    mem_out = B * N * d_model

    mem_total_layer = mem_input + mem_qkv + mem_scores + mem_attn + mem_heads + mem_out

    # double si backward
    if training:
        mem_total_layer *= 2

    # total pour tous les blocs
    mem_total = mem_total_layer * num_blocks

    # bytes → MB
    return mem_total * bytes_per_float / (1024 ** 2)

def compute_weight_memory(model):
    total_bytes = 0
    for p in model.parameters():
        total_bytes += p.numel() * p.element_size()
    return total_bytes / (1024**2)


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
    except:
        auc = float("nan")

    return acc, f1, prec, rec, auc

# =========================
# Transformer Block STANDARD
# =========================
class StandardBlock(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()

        self.attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            batch_first=True
        )

        self.norm1 = nn.LayerNorm(d_model)

        self.ff = nn.Sequential(
            nn.Linear(d_model, 4*d_model),
            nn.ReLU(),
            nn.Linear(4*d_model, d_model)
        )

        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x):
        attn_out, _ = self.attn(x, x, x)
        x = self.norm1(x + attn_out)

        ff_out = self.ff(x)
        x = self.norm2(x + ff_out)

        return x

# =========================
# Positional Encoding
# =========================
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super().__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)

        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0)/d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.pe = pe.unsqueeze(0)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)].to(x.device)

# =========================
# BASELINE MODEL
# =========================
class BaselineClassifier(nn.Module):
    def __init__(self, vocab_size, num_classes):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, D_MODEL)
        self.pos_encoding = PositionalEncoding(D_MODEL)

        self.blocks = nn.ModuleList([
            StandardBlock(D_MODEL, NUM_HEADS) for _ in range(NUM_BLOCKS)
        ])

        self.classifier = nn.Linear(D_MODEL, num_classes)

    def forward(self, input_ids, attention_mask=None):
        x = self.embedding(input_ids)
        x = self.pos_encoding(x)

        for block in self.blocks:
            x = block(x)

        pooled = x.mean(dim=1)
        logits = self.classifier(pooled)

        return logits
"""
# =========================
# DATA LOADING (IDENTIQUE)
# =========================
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
"""

# =========================================================
# Balanced subset par classe
# =========================================================
def get_balanced_subset(dataset, label_col="label", ratio=1.0, seed=42):
    if ratio >= 1.0:
        return dataset

    np.random.seed(seed)

    class_indices = defaultdict(list)

    for i, example in enumerate(dataset):
        label = example[label_col]
        class_indices[label].append(i)

    selected_indices = []

    for label, indices in class_indices.items():
        n = int(len(indices) * ratio)
        n = max(n, 1)  # au moins 1 sample

        chosen = np.random.choice(indices, n, replace=False)
        selected_indices.extend(chosen)

    return dataset.select(selected_indices)


# =========================================================
# Load data
# =========================================================
def load_data(dataset_name="snli",
              batch_size=32,
              max_length=128,
              subset_ratio=1.0):

    print(f"Chargement du dataset {dataset_name}...")
    dataset = load_dataset(dataset_name)

    print("Chargement du tokenizer bert-base-uncased...")
    tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-uncased")

    dataset = dataset.shuffle(seed=42)

    # =========================
    # SNLI
    # =========================
    if dataset_name == "snli":

        dataset = dataset.filter(lambda example: example['label'] != -1)

        train_dataset = get_balanced_subset(
            dataset["train"],
            ratio=subset_ratio
        )

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

    # =========================
    # Classification datasets
    # =========================
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

        train_dataset = get_balanced_subset(
            dataset["train"],
            ratio=subset_ratio
        )

        if "validation" in dataset:
            val_dataset = dataset["validation"]
        else:
            val_dataset = dataset["test"]

        if "test" in dataset:
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
    # Logs utiles
    # =========================
    print(f"Subset ratio: {subset_ratio}")
    print(f"Train size: {len(train_dataset)}")
    print(f"Validation size: {len(val_dataset)}")
    print(f"Test size: {len(test_dataset)}")

    # =========================
    # Tokenisation
    # =========================
    print("Tokenisation en cours...")

    train_dataset = train_dataset.map(tokenize_function, batched=True)
    val_dataset = val_dataset.map(tokenize_function, batched=True)
    test_dataset = test_dataset.map(tokenize_function, batched=True)

    # =========================
    # Format PyTorch
    # =========================
    train_dataset.set_format(
        type='torch',
        columns=['input_ids', 'attention_mask', 'label']
    )

    val_dataset.set_format(
        type='torch',
        columns=['input_ids', 'attention_mask', 'label']
    )

    test_dataset.set_format(
        type='torch',
        columns=['input_ids', 'attention_mask', 'label']
    )

    # =========================
    # DataLoaders
    # =========================
    train_dataloader = DataLoader(
        train_dataset,
        shuffle=True,
        batch_size=batch_size
    )

    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size
    )

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=batch_size
    )

    return (
        train_dataloader,
        val_dataloader,
        test_dataloader,
        tokenizer.vocab_size,
        num_classes
    )



# =========================
# TRAINING
# =========================
#train_loader, val_loader, test_loader, vocab_size, num_classes = load_data("dbpedia_14", BATCH_SIZE, 64)
#train_loader, val_loader, test_loader, vocab_size, num_classes = load_data("ag_news", BATCH_SIZE, 64)
train_loader, val_loader, test_loader, vocab_size, num_classes = load_data("snli", BATCH_SIZE, 64, subset_ratio=0.5)
#train_loader, val_loader, test_loader, vocab_size, num_classes = load_data("imdb", BATCH_SIZE, 512)
#train_loader, val_loader, test_loader, vocab_size, num_classes = load_data("yelp_review_full", BATCH_SIZE, 512)

model = BaselineClassifier(vocab_size, num_classes).to(DEVICE)
optimizer = AdamW(model.parameters(), lr=LR)

total_steps = NUM_EPOCHS * len(train_loader)
step = 0
best_val_acc = 0.0
total_carbon = 0.0

weight_memory = compute_weight_memory(model)
print(f"Model weight memory: {weight_memory:.2f} MB")

for epoch in range(1, NUM_EPOCHS+1):
    model.train()

    train_preds, train_true, train_probs = [], [], []
    train_flops_sum, train_carbon_sum, train_mem_sum = 0, 0, 0
    train_peak_mem_sum, train_batches = 0, 0

    for batch in train_loader:
        step += 1
        train_batches += 1

        ids = batch["input_ids"].to(DEVICE)
        labels = batch["label"].to(DEVICE)

        torch.cuda.reset_peak_memory_stats(DEVICE)

        logits = model(ids)

        peak_mem = torch.cuda.max_memory_allocated(DEVICE)/(1024**2)

        seq_len = ids.shape[1]
        batch_size = ids.shape[0]

        flops = compute_flops(seq_len, batch_size, D_MODEL, NUM_HEADS, NUM_BLOCKS)
        carbon = compute_carbon(flops)
        mem = compute_activation_memory(ids, D_MODEL, NUM_HEADS, NUM_BLOCKS)

        train_flops_sum += flops
        train_carbon_sum += carbon
        train_mem_sum += mem
        train_peak_mem_sum += peak_mem
        total_carbon += carbon

        probs = F.softmax(logits, dim=-1)
        preds = logits.argmax(dim=-1)

        train_preds.extend(preds.detach().cpu().numpy())
        train_true.extend(labels.detach().cpu().numpy())
        train_probs.extend(probs.detach().cpu().numpy())

        loss = F.cross_entropy(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        is_last_step = (epoch == NUM_EPOCHS and step == total_steps)

        if (step % 300 == 0) or is_last_step:

            train_probs_np = np.array(train_probs)
            acc, f1, prec, rec, auc = compute_metrics(train_preds, train_true, train_probs_np)

            print(f"\nStep {step}")
            print(f"Loss {loss.item():.4f}")

            print(f"TRAIN → Acc {acc:.3f} | F1 {f1:.3f} | Prec {prec:.3f} | "
                  f"Recall {rec:.3f} | AUC {auc:.3f}")

            print(f"TRAIN batches: {train_batches}")
            print(f"TRAIN Total FLOPs: {train_flops_sum:.2e}")
            print(f"TRAIN Total Carbon (gCO2): {train_carbon_sum:.6f}")
            print(f"TRAIN Avg Activation memory (MB): {train_mem_sum/train_batches:.2f}")
            print(f"TRAIN Peak GPU memory (MB): {train_peak_mem_sum/train_batches:.2f}")
            print(f"TRAIN Weight memory (MB): {weight_memory:.2f}")

            # RESET
            train_preds, train_true, train_probs = [], [], []
            train_flops_sum, train_carbon_sum, train_mem_sum = 0,0,0
            train_peak_mem_sum, train_batches = 0,0

            # VALIDATION
            model.eval()

            val_preds, val_true, val_probs = [], [], []
            val_loss = 0

            # accumulateurs validation
            val_flops, val_carbon, val_mem, val_peak_mem_sum = 0, 0, 0, 0
            val_batches = 0

            with torch.no_grad():
                for vbatch in val_loader:
                    val_batches += 1

                    vid = vbatch["input_ids"].to(DEVICE)
                    vlabels = vbatch["label"].to(DEVICE)

                    torch.cuda.reset_peak_memory_stats(DEVICE)

                    v_logits = model(vid)

                    v_peak_mem_batch = torch.cuda.max_memory_allocated(DEVICE) / (1024 ** 2)

                    # ===== METRICS =====
                    v_seq_len = vid.shape[1]
                    v_batch_size = vid.shape[0]

                    v_flops_batch = compute_flops(v_seq_len, v_batch_size, D_MODEL, NUM_HEADS, NUM_BLOCKS)
                    v_carbon_batch = compute_carbon(v_flops_batch)
                    v_activation_memory = compute_activation_memory(vid, D_MODEL, NUM_HEADS, NUM_BLOCKS)

                    val_flops += v_flops_batch
                    val_carbon += v_carbon_batch
                    val_mem += v_activation_memory
                    val_peak_mem_sum += v_peak_mem_batch

                    v_probs = F.softmax(v_logits, dim=-1)
                    v_preds = v_logits.argmax(dim=-1)

                    val_preds.extend(v_preds.detach().cpu().numpy())
                    val_true.extend(vlabels.detach().cpu().numpy())
                    val_probs.extend(v_probs.detach().cpu().numpy())

                    val_loss += F.cross_entropy(v_logits, vlabels).item()

            val_probs_np = np.array(val_probs)

            val_acc, val_f1, val_prec, val_rec, val_auc = compute_metrics(
                val_preds, val_true, val_probs_np
            )

            print(f"\nVAL → Loss {val_loss / len(val_loader):.4f} | "
                  f"Acc {val_acc:.3f} | F1 {val_f1:.3f} | "
                  f"Prec {val_prec:.3f} | Recall {val_rec:.3f} | "
                  f"AUC {val_auc:.3f}")

            # ✅ AJOUT IMPORTANT
            print(f"VAL batches: {val_batches}")
            print(f"VAL Total FLOPs: {val_flops:.2e}")
            print(f"VAL Total Carbon (gCO2): {val_carbon:.6f}")
            print(f"VAL Avg Activation memory (MB): {val_mem / val_batches:.2f}")
            print(f"VAL Peak GPU memory (MB): {val_peak_mem_sum / val_batches:.2f}")

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(model.state_dict(), "baseline_best.pt")
                print("NEW BEST MODEL SAVED")

            model.train()

torch.save(model.state_dict(), "baseline_final.pt")

print(f"Training finished. Best val acc: {best_val_acc:.4f}. Total carbon Train {total_carbon:.4f} gCO2")

# =========================
# TEST
# =========================
print("\n===== TEST =====")

model.eval()

all_preds, all_true, all_probs = [], [], []

# accumulateurs test
test_flops, test_carbon, test_mem, test_peak_mem_sum = 0, 0, 0, 0
test_batches = 0

with torch.no_grad():
    for batch in test_loader:
        test_batches += 1

        ids = batch["input_ids"].to(DEVICE)
        labels = batch["label"].to(DEVICE)

        torch.cuda.reset_peak_memory_stats(DEVICE)

        logits = model(ids)

        peak_mem_batch = torch.cuda.max_memory_allocated(DEVICE) / (1024**2)

        # ===== METRICS =====
        seq_len = ids.shape[1]
        batch_size = ids.shape[0]

        flops_batch = compute_flops(seq_len, batch_size, D_MODEL, NUM_HEADS, NUM_BLOCKS)
        carbon_batch = compute_carbon(flops_batch)
        activation_memory = compute_activation_memory(ids, D_MODEL, NUM_HEADS, NUM_BLOCKS, training=False)

        test_flops += flops_batch
        test_carbon += carbon_batch
        test_mem += activation_memory
        test_peak_mem_sum += peak_mem_batch

        probs = F.softmax(logits, dim=-1)
        preds = logits.argmax(dim=-1)

        all_preds.extend(preds.detach().cpu().numpy())
        all_true.extend(labels.detach().cpu().numpy())
        all_probs.extend(probs.detach().cpu().numpy())

all_probs = np.array(all_probs)

acc, f1, prec, rec, auc = compute_metrics(all_preds, all_true, all_probs)

print(f"\nTEST → Acc {acc:.4f} | F1 {f1:.4f} | "
      f"Prec {prec:.4f} | Recall {rec:.4f} | AUC {auc:.4f}")

# ✅ AJOUT IMPORTANT
print(f"TEST batches: {test_batches}")
print(f"TEST Total FLOPs: {test_flops:.2e}")
print(f"TEST Total Carbon (gCO2): {test_carbon:.6f}")
print(f"TEST Avg Activation memory (MB): {test_mem/test_batches:.2f}")
print(f"TEST Peak GPU memory (MB): {test_peak_mem_sum/test_batches:.2f}")