import math
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from datasets import load_dataset, DatasetDict
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from torch.optim import AdamW
from nltk.translate.bleu_score import sentence_bleu
from rouge_score import rouge_scorer
from bert_score import score as bertscore

# =========================
# CONFIG
# =========================
device = "cuda" if torch.cuda.is_available() else "cpu"
batch_size = 16
max_len = 1024
epochs = 3
lr = 5e-4
D_MODEL = 768
NUM_HEADS = 12
NUM_BLOCKS = 12


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
        shift_labels.view(-1),
        ignore_index=-100
    )

def evaluate_ppl(model, val_loader, device):

    model.eval()

    total_loss = 0.0
    n_batches = 0
    val_flops, val_carbon, val_mem, val_peak_mem_sum = 0, 0, 0, 0

    with torch.no_grad():

        for batch in val_loader:

            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            torch.cuda.reset_peak_memory_stats(device)
            logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
            v_peak_mem_batch = torch.cuda.max_memory_allocated(device) / (1024 ** 2)

            v_seq_len = logits.shape[1]
            v_batch_size = logits.shape[0]
            v_flops_batch = compute_flops(v_seq_len, v_batch_size, D_MODEL, NUM_HEADS, NUM_BLOCKS)
            v_carbon_batch = compute_carbon(v_flops_batch)
            v_activation_memory = compute_activation_memory(logits, D_MODEL, NUM_HEADS, NUM_BLOCKS)

            val_flops += v_flops_batch
            val_carbon += v_carbon_batch
            val_mem += v_activation_memory
            val_peak_mem_sum += v_peak_mem_batch

            loss = compute_loss(logits, labels)

            total_loss += loss.item()
            n_batches += 1

    avg_loss = total_loss / n_batches
    ppl = math.exp(avg_loss)

    return avg_loss, ppl, val_flops, val_carbon, val_mem, val_peak_mem_sum, n_batches

# =========================
# GENERATION (EOS STOP ✔)
# =========================
def generate(model, tokenizer, instruction, inp="<noinput>", device="cuda",
             eos_id=None, max_new_tokens=100, temperature=1.0, top_k=50):

    prompt = f"Instruction: {instruction}\nInput: {inp}\nResponse:"
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)

    with torch.no_grad():
        for _ in range(max_new_tokens):

            logits = model(input_ids=input_ids).logits[:, -1, :]

            # 🔥 temperature
            logits = logits / temperature

            # 🔥 top-k filtering
            if top_k is not None:
                values, indices = torch.topk(logits, top_k)
                probs = torch.softmax(values, dim=-1)
                next_token = indices[0, torch.multinomial(probs, 1)]
            else:
                probs = torch.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, 1)

            # STOP CONDITION
            if eos_id is not None and next_token.item() == eos_id:
                break

            input_ids = torch.cat([input_ids, next_token.unsqueeze(0)], dim=1)

    return tokenizer.decode(input_ids[0], skip_special_tokens=True)

# =========================
# METRICS
# =========================
rouge = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)

def evaluate_generation(model, tokenizer, dataset, eval_gen_samples=300, device="cuda"):

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

        pred = generate(model, tokenizer, instruction, inp, eos_id=tokenizer.eos_id, device=device)

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
# ========================
# METRICS
# =======================
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


def compute_activation_memory(logits, d_model, num_heads, num_blocks, training=True):
    """
    ids: (B, N)
    """
    B, N, _ = logits.shape
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
        test_size=5000, # 5%
        seed=42
    )
    dataset = DatasetDict({
        "train": split["train"],
        "validation": split["test"]
    })

    dataset = dataset.map(format_example)
    dataset = dataset.filter(lambda x: x["instruction"] is not None and len(x["instruction"]) > 0 and x["output"] is not None and len(x["output"]) > 0)

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
    model = GPT2LMHeadModel.from_pretrained("gpt2").to(device)
    optimizer = AdamW(model.parameters(), lr=lr)

    best_val_loss = float("inf")
    global_step = 0
    patience = 3
    patience_counter = 0
    early_stop = False

    steps_per_epoch = len(train_loader)
    total_steps = epochs * steps_per_epoch
    log_every = max(1, total_steps // 100)

    print("Steps per epoch:", steps_per_epoch)
    print("Total steps:", total_steps)
    print("Log every:", log_every)

    weight_memory = compute_weight_memory(model)
    print(f"Model weight memory: {weight_memory:.2f} MB")

    total_carbon = 0.0

    for epoch in range(epochs):
        train_flops_sum, train_carbon_sum, train_mem_sum = 0, 0, 0
        train_peak_mem_sum, train_batches = 0, 0
        model.train()
        optimizer.zero_grad()

        for batch in train_loader:

            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            attention_mask = batch['attention_mask'].to(device)
            torch.cuda.reset_peak_memory_stats(device)
            logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
            peak_mem = torch.cuda.max_memory_allocated(device) / (1024 ** 2)

            loss = compute_loss(logits, labels)

            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()
            optimizer.zero_grad()

            seq_len = logits.shape[1]
            batch_size = logits.shape[0]
            flops = compute_flops(seq_len, batch_size, D_MODEL, NUM_HEADS, NUM_BLOCKS)
            carbon = compute_carbon(flops)
            mem = compute_activation_memory(logits, D_MODEL, NUM_HEADS, NUM_BLOCKS)

            train_flops_sum += flops
            train_carbon_sum += carbon
            train_mem_sum += mem
            train_peak_mem_sum += peak_mem
            total_carbon += carbon

            global_step += 1

            # =========================
            # LOG + EVAL
            # =========================
            if global_step % log_every == 0:
                print(f"\n[STEP {global_step}]")
                print(f"Train Loss: {loss.item():.4f}")
                print(f"TRAIN batches: {train_batches}")
                print(f"TRAIN Total FLOPs: {train_flops_sum:.2e}")
                print(f"TRAIN Total Carbon (gCO2): {train_carbon_sum:.6f}")
                print(f"TRAIN Avg Activation memory (MB): {train_mem_sum / train_batches:.2f}")
                print(f"TRAIN Peak GPU memory (MB): {train_peak_mem_sum / train_batches:.2f}")
                print(f"TRAIN Weight memory (MB): {weight_memory:.2f}")
                val_loss, val_ppl, val_flops, val_carbon, val_mem, val_peak_mem_sum, n_batches = evaluate_ppl(model, val_loader, device)
                print(f"VAL batches: {n_batches} | Val Loss  : {val_loss:.4f} | Val PPL   : {val_ppl:.2f}")
                print(f"VAL Total FLOPs: {val_flops:.2e}")
                print(f"VAL Total Carbon (gCO2): {val_carbon:.6f}")
                print(f"VAL Avg Activation memory (MB): {val_mem / n_batches:.2f}")
                print(f"VAL Peak GPU memory (MB): {val_peak_mem_sum / n_batches:.2f}")
                bleu, rouge, bert_score = evaluate_generation(model, tokenizer, dataset, eval_gen_samples=300)
                print(f"Bleu: {bleu:.4f} | Rouge L: {rouge:.4f} | BER score: {bert_score:.4f}")

                train_flops_sum, train_carbon_sum, train_mem_sum = 0, 0, 0
                train_peak_mem_sum, train_batches = 0, 0

                # save best model
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    torch.save(model.state_dict(), "model.pt")
                    print("✅ Best model saved")


                """else:
                    patience_counter += 1
                    print(f"⏳ Patience: {patience_counter}/{patience}")

                    if patience_counter >= patience:
                        print("🛑 Early stopping triggered")
                        early_stop = True
                        break"""

        print(f"Epoch {epoch + 1} done")
        """if early_stop:
            break"""

    # =========================
    # FINAL EVALUATION
    # =========================
    print("\n🔍 FINAL EVALUATION")

    val_loss, val_ppl, val_flops, val_carbon, val_mem, val_peak_mem_sum, n_batches = evaluate_ppl(model, val_loader, device)
    bleu, rouge, bert_score = evaluate_generation(model, tokenizer, dataset, eval_gen_samples=300)
    print("===== FINAL RESULTS =====")
    print(f"Total carbon Train {total_carbon:.4f} gCO2")
    print(f"VAL batches: {n_batches} | Val Loss  : {val_loss:.4f} | Val PPL   : {val_ppl:.2f}")
    print(f"VAL Total FLOPs: {val_flops:.2e}")
    print(f"VAL Total Carbon (gCO2): {val_carbon:.6f}")
    print(f"VAL Avg Activation memory (MB): {val_mem / n_batches:.2f}")
    print(f"VAL Peak GPU memory (MB): {val_peak_mem_sum / n_batches:.2f}")
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

    for i, (instruction, inp) in enumerate(prompts):
        output = generate(
            model=model,
            tokenizer=tokenizer,
            instruction=instruction,
            inp=inp,
            device=device,
            eos_id=tokenizer.eos_token_id,
            max_new_tokens=100,
            temperature=1.0,
            top_k=50
        )

        print(f"\n--- SAMPLE {i + 1} ---")
        print(f"INSTRUCTION: {instruction}")
        print(f"INPUT: {inp}")
        print(f"OUTPUT:\n{output}")

