import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from transformers import  AutoTokenizer, LongformerModel
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


####################################################
# Question Answering (QA) Model
####################################################
class MyLongFormerForQA(nn.Module):
    def __init__(self, dropout_prob=0.2):
        super().__init__()

        self.model = LongformerModel.from_pretrained("allenai/longformer-base-4096")

        hidden = self.model.config.hidden_size

        self.qa_outputs = nn.Linear(hidden, 2)  # start + end
        self.dropout = nn.Dropout(dropout_prob)

    def forward(
        self,
        input_ids,
        attention_mask=None,
        global_attention_mask=None,
    ):
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            global_attention_mask=global_attention_mask
        )

        sequence_output = outputs["last_hidden_state"]  # (B, T, H)
        sequence_output = self.dropout(sequence_output)

        logits = self.qa_outputs(sequence_output)  # (B, T, 2)

        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)


        return {
            "start_logits": start_logits,
            "end_logits": end_logits,
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
# METRICS
# =========================
def compute_flops(
    seq_len,
    batch_size,
    d_model,
    num_heads,
    num_blocks,
    window_size=512,
    N_global=1
):
    B = batch_size
    N = seq_len
    H = num_heads
    d_head = d_model // num_heads
    L = num_blocks
    w = window_size
    g = N_global

    # 1. QKV projections (local + global)
    flops_qkv = 3 * B * N * d_model * d_model
    flops_qkv_global = 3 * B * g * d_model * d_model

    # 2. Local attention (QK^T + AV)
    flops_local_attn = 2 * B * H * N * w * d_head

    # 3. Global attention (QK^T + AV)
    flops_global_attn = 2 * B * H * N * g * d_head

    # 4. Output projection
    flops_out = B * N * d_model * d_model

    # Total per layer
    flops_per_layer = (
        flops_qkv
        + flops_qkv_global
        + flops_local_attn
        + flops_global_attn
        + flops_out
    )

    return L * flops_per_layer

def compute_activation_memory(
    ids,
    d_model,
    num_heads,
    num_blocks,
    window_size=512,
    N_global=1,
    training=True
):
    B, N = ids.shape
    H = num_heads
    bytes_per_float = 4

    # 1. Input
    mem_input = B * N * d_model

    # 2. QKV
    mem_qkv = 3 * B * N * d_model
    mem_qkv_global = 3 * B * N_global * d_model

    # 3. Local attention
    mem_scores_local = B * H * N * window_size
    mem_attn_local = B * H * N * window_size

    # 4. Global attention
    mem_scores_global = B * H * N * N_global
    mem_attn_global = B * H * N * N_global

    # 5. Heads + output
    mem_heads = B * N * d_model
    mem_out = B * N * d_model

    mem_total_layer = (
        mem_input
        + mem_qkv + mem_qkv_global
        + mem_scores_local + mem_attn_local
        + mem_scores_global + mem_attn_global
        + mem_heads + mem_out
    )

    if training:
        mem_total_layer *= 2

    mem_total = mem_total_layer * num_blocks

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
def run_validation(model, val_loader,  d_model=768, num_heads=8, training=True, num_blocks=12):
    model.eval()

    val_start_preds, val_end_preds = [], []
    val_start_true, val_end_true = [], []


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
            # ✅ global attention mask
            global_attention_mask = torch.zeros_like(vmask)
            global_attention_mask[:, 0] = 1
            global_attention_mask = global_attention_mask.to(DEVICE)

            v_outputs = model(
                vid,
                vmask,
                global_attention_mask
            )

            v_start_logits = v_outputs["start_logits"]
            v_end_logits = v_outputs["end_logits"]

            # =========================
            # METRICS
            # =========================
            v_seq_len = vid.shape[1]
            v_batch_size = vid.shape[0]

            v_flops_batch = compute_flops(
                v_seq_len, v_batch_size, d_model=d_model, num_heads=num_heads, num_blocks=num_blocks
            )
            v_carbon_batch = compute_carbon(v_flops_batch)
            v_activation_memory = compute_activation_memory(
                vid, d_model=d_model, num_heads=num_heads, training=training, num_blocks=num_blocks
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

            # =========================
            # LOSS
            # =========================
            v_L = (
                F.cross_entropy(v_start_logits, v_start) +
                F.cross_entropy(v_end_logits, v_end)
            ) / 2



            val_loss += v_L.item()

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
        "flops": val_flops,
        "carbon": val_carbon,
        "mem": val_mem / val_batches,
    }

#######################################
# Training
#######################################

if __name__ == "__main__":
    model = MyLongFormerForQA()

    optimizer = AdamW(model.parameters(), lr=LR)
    total_steps = NUM_EPOCHS * len(train_loader)
    step = 0
    best_train_f1 = 0.0
    best_train_em = 0.0
    total_flops, total_carbon = 0, 0
    D_MODEL = model.model.config.hidden_size
    NUM_HEADS = model.model.config.num_attention_heads
    NUM_BLOCKS = model.model.config.num_hidden_layers
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

            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            start_pos = batch["start_positions"].to(DEVICE)
            end_pos = batch["end_positions"].to(DEVICE)
            global_attention_mask = torch.zeros_like(attention_mask, device=attention_mask.device)
            global_attention_mask[:, 0] = 1


            torch.cuda.reset_peak_memory_stats(DEVICE)
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                global_attention_mask=global_attention_mask

            )
            peak_mem_batch = torch.cuda.max_memory_allocated(DEVICE) / (1024**2)

            start_logits = outputs["start_logits"]
            end_logits = outputs["end_logits"]

            # =========================
            # METRICS COMPUTE
            # =========================
            seq_len = input_ids.shape[1]
            batch_size = input_ids.shape[0]

            flops_batch = compute_flops(seq_len, batch_size, D_MODEL, NUM_HEADS, NUM_BLOCKS)
            carbon_batch = compute_carbon(flops_batch)
            activation_memory = compute_activation_memory(input_ids, D_MODEL, NUM_HEADS, NUM_BLOCKS)

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

            # =========================
            # LOSSES
            # =========================
            L_start = F.cross_entropy(start_logits, start_pos)
            L_end = F.cross_entropy(end_logits, end_pos)
            L_task = (L_start + L_end) / 2


            loss = L_task

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
                print(f"Loss {loss.item():.4f} | L_task {L_task.item():.4f} ")

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
    print("STARTING TEST PHASE")
    print("==============================\n")
    val_metrics = run_validation(model, val_loader, training=False, d_model=D_MODEL, num_heads=NUM_HEADS, num_blocks=NUM_BLOCKS)
    print(
        f"VAL Loss {val_metrics['loss']:.4f} | EM {val_metrics['em']:.3f} | F1 {val_metrics['f1']:.3f} | s_mean {val_metrics['s_mean']:.3f}")
    print(f"VAL Total FLOPs: {val_metrics['flops']:.2e}")
    print(f"VAL Carbon: {val_metrics['carbon']:.6f} gCO2")
    print(f"VAL Mem: {val_metrics['mem']:.2f} MB")

