# BudgetFormer: Adaptive Head Budgeting for Efficient Multi-Head Attention

BudgetFormer is an adaptive framework that dynamically allocates the number of active attention heads on a per-input basis. By learning both an input-dependent head budget and a relevance distribution over heads, BudgetFormer reduces theoretical inference computational complexity (FLOPs), activation memory, and estimated carbon emissions while matching or surpassing the predictive performance of standard dense multi-head attention.

## Key Features

- **Dynamic Head Budget Allocation**: Learns an input-dependent head budget ($s_{\text{pred}}$) and head relevance distribution ($q_{\text{head}}$), activating only the most informative heads per sample.
- **Over-Parameterization Mitigation**: Eliminates redundant attention heads dynamically, showing that optimal performance is frequently achieved with far fewer active heads than standard configurations.
- **Broad Backbone Support**: Implemented for both from-scratch encoders/decoders and standard pre-trained backbones (BERT, RoBERTa, DistilBERT, DeBERTa, Longformer).
- **Efficiency and Green AI Profiling**: Built-in estimators for theoretical inference FLOPs, activation memory, parameter memory, and carbon emission approximations (gCO2).
- **Exploration-Exploitation Training Strategy**: Balances exploratory gating during training (temperature, noise, and entropy scheduling) with focused budget utilization.

## Quickstart

### Installation

```bash
conda create -n BudgetFormer python=3.10 pip -y
conda activate BudgetFormer
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
pip install transformers datasets scikit-learn numpy
```

### Datasets

The repository natively supports standard Hugging Face datasets:

- **DBpedia-14**: [dbpedia_14](https://huggingface.co/datasets/dbpedia_14) - Large-scale topic categorization (14 classes)
- **AG News**: [ag_news](https://huggingface.co/datasets/ag_news) - Topic classification (4 classes)
- **IMDb**: [imdb](https://huggingface.co/datasets/imdb) - Binary sentiment classification
- **SNLI**: [snli](https://huggingface.co/datasets/snli) - Natural language inference (3 classes)
- **Yelp Review Full**: [yelp_review_full](https://huggingface.co/datasets/yelp_review_full) - Fine-grained review sentiment (5 classes)

## Usage

### 1. Proof of Concept (POC) & Ablations

Train the baseline Transformer:
```bash
python "AdaTransformer/1. POC/baseline.py"
```

Train BudgetFormer with dynamic head allocation:
```bash
python "AdaTransformer/1. POC/adaT.py"
```

Run ablation studies (Fixed budgets, Random selection, Gating variants):
```bash
python "AdaTransformer/1. POC/ablation.py"
```

### 2. Pretrained Encoders

BudgetFormer is adapted across various transformer encoder backbones:

- **BERT**:
  ```bash
  python "AdaTransformer/2. Encoder/BERT/adaT.py"
  ```
- **RoBERTa**:
  ```bash
  python "AdaTransformer/2. Encoder/Roberta/adaT.py"
  ```
- **DistilBERT**:
  ```bash
  python "AdaTransformer/2. Encoder/DistilBERT/adaT.py"
  ```
- **DeBERTa**:
  ```bash
  python "AdaTransformer/2. Encoder/Deberta/adaT.py"
  ```
- **Longformer**:
  ```bash
  python "AdaTransformer/2. Encoder/LongFormer/adaT.py"
  ```

### 3. Decoder Models

Test autoregressive and decoder architectures with KV-cache support:

```bash
# In-distribution training
python "AdaTransformer/3. Deocder/adaT/scratch.py"

# Out-of-distribution (OOD) evaluation
python "AdaTransformer/3. Deocder/adaT/ood.py"

# Instruction fine-tuning
python "AdaTransformer/3. Deocder/adaT/instruct.py"
```

### 4. Task-Specific Benchmarks

Run dedicated task experiments:

```bash
cd "AdaTransformer/ada_transformer/adaT/snli"
python main.py
```

## Experimental Results

Below are the experimental results reported in the submission paper:

### Main Results

| Dataset | Standard Transformer Acc | BudgetFormer Acc | Transformer FLOPs ($10^{12}$) | BudgetFormer FLOPs ($10^{12}$) | Transformer Est. CO2 (g) | BudgetFormer Est. CO2 (g) | Mean Budget ($s_{\text{mean}}$) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **DBpedia** | 0.9830 | **0.9859** | 44.0 | **42.6** | 0.1468 | **0.1420** | 0.085 |
| **AG News** | **0.9099** | 0.9022 | 4.78 | **4.65** | 0.0159 | **0.0155** | 0.212 |
| **IMDB** | 0.8354 | **0.8356** | 161 | **145** | 0.5376 | **0.4840** | 0.601 |
| **SNLI** | 0.7835 | **0.8106** | 6.18 | **6.05** | 0.0206 | **0.0202** | 0.364 |
| **Yelp** | 0.5810 | **0.6190** | 323 | **258** | 1.0751 | **0.8591** | 0.198 |

*FLOPs are reported in units of $10^{12}$ for all datasets. $s_{\text{mean}}$ represents the average predicted fraction of attention heads allocated per input. Carbon emission figures are theoretical estimates derived from FLOPs count and GPU power models (not direct physical measurements).*

### Ablation Study

Ablation evaluation comparing fixed budgets ($s \in \{0.1, 0.25, 0.5, 0.75, 1.0\}$) without the learned budget predictor and random head selection:

| Dataset | $s=0.1$ | $s=0.25$ | $s=0.5$ | $s=0.75$ | $s=1.0$ | Random Selection | Full BudgetFormer |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **DBpedia** | 0.9846 | 0.9478 | 0.6202 | 0.3245 | 0.1670 | 0.7511 | **0.9859** |
| **AG News** | 0.8905 | 0.8974 | 0.8675 | 0.8139 | 0.7554 | 0.8180 | **0.9022** |
| **IMDB** | 0.7296 | 0.8036 | 0.8246 | 0.8289 | 0.8162 | 0.7493 | **0.8356** |
| **SNLI** | 0.6704 | 0.7598 | 0.5838 | 0.3541 | 0.3190 | 0.3370 | **0.8106** |
| **Yelp** | 0.5818 | 0.5813 | 0.2737 | 0.1013 | 0.0743 | 0.3244 | **0.6190** |

## Implementation & Methodology Notes

### 1. Architectural Configuration (Paper Results vs. POC Exploration)
- **Paper Benchmark Configuration**: The official benchmark results in the paper were produced using **4 layers (blocks)** and **8 attention heads** with hidden dimension $d_{\text{model}} = 768$ (as specified in `AdaTransformer/ada_transformer/adaT/*/main.py`).
- **Exploratory POC Configuration**: The proof-of-concept script (`AdaTransformer/1. POC/adaT.py`) contains a scaled configuration with **6 layers and 12 heads** used for larger-scale architecture exploration.

### 2. Entropy Schedule Alignment ($\beta_{\text{schedule}}$)
In the training loss:
$$\mathcal{L} = \mathcal{L}_{\text{task}} + \mathcal{L}_{\text{budget}} + \beta(t) \mathcal{L}_{\text{entropy}}$$
The implemented schedule in `beta_schedule(step, total_steps)` is:
$$\beta(t) = \beta_{\max} \left(1 - \frac{2t}{T}\right)$$
This coefficient transitions from $+\beta_{\max}$ down to $-\beta_{\max}$. At the start of training ($t \to 0$), a positive coefficient maximizes head entropy to encourage broad exploration across all attention heads. As training progresses ($t \to T$), the coefficient becomes negative, penalizing dispersion and encouraging sparse head specialization. *(Note: This aligns the practical implementation with the exploration-to-exploitation mechanism).*

### 3. Theoretical FLOPs vs. Wall-Clock Latency
The FLOPs reductions and memory savings reported in the paper represent **theoretical complexity reductions under sparse execution**. In standard PyTorch without custom sparse CUDA kernels, full attention projections and scores are computed synchronously across all heads before the Top-K gating mask is applied. Therefore, these savings should be interpreted as theoretical computational bounds rather than measured wall-clock acceleration on dense GPU hardware.

### 4. Carbon Footprint Estimations
Reported CO2 emissions are **theoretical estimates** modeled based on the computed FLOPs, assumed GPU hardware throughput (e.g., NVIDIA A100 TDP of 300W), and a standard carbon intensity factor ($0.4 \text{ kg CO}_2/\text{kWh}$), rather than direct hardware wattmeter measurements.

## Empirical Visualizations & Analysis

### 1. Predicted Budget Distributions Across Benchmarks
The distribution of predicted budgets reflects individual task complexity. Coarser-grained tasks (such as DBpedia-14) converge to small head budgets ($s_{\text{mean}} = 0.085$), whereas richer semantics (IMDb, SNLI) adaptively allocate larger budgets.

<p align="center">
  <img src="assets/images/all_boxplots.png" alt="Budget Distributions Across Tasks" width="80%">
</p>

### 2. Multi-Head Sparsity and Performance Curves
Evaluating accuracy across head budgets demonstrates that standard full-head allocation is largely over-parameterized. Peak performance is achieved with a fraction of active heads:

<p align="center">
  <img src="assets/images/all_lines.png" alt="Accuracy across head budgets" width="90%">
</p>

### 3. Task Complexity & Semantic Ambiguity (SNLI Analysis)
On SNLI, neutral pairs (semantically ambiguous) exhibit higher mean budgets ($s_{\text{mean}}$) and greater entropy than straightforward entailment pairs, showing that BudgetFormer dynamically expands its attention capacity when facing harder inputs:

<p align="center">
  <img src="assets/images/snli_smean.png" alt="SNLI Mean Budget" width="31%">
  <img src="assets/images/snli_std.png" alt="SNLI Budget Variance" width="31%">
  <img src="assets/images/snli_entropy.png" alt="SNLI Head Selection Entropy" width="31%">
</p>

### 4. Attention Allocation & Head Specialization Heatmaps
Sample-level visualization of attention scores and gating weights demonstrates layer-wise sparsity and head specialization:

<p align="center">
  <img src="assets/images/attention_dbpedia_example.png" alt="Attention Allocation Heatmap" width="65%">
</p>

## Interactive Notebook

An end-to-end interactive demo containing model definition, training loops, profiling (FLOPs, memory, carbon), and attention head visualization heatmaps is available at:
- `AdaTransformer/AdaTransformer.ipynb`

## Default Training Configuration

- **Benchmark Architecture**: 4 Layers, 8 Attention Heads, $d_{\text{model}} = 768$
- **POC Exploration**: 6 Layers, 12 Attention Heads, $d_{\text{model}} = 768$
- **Batch Size**: 16
- **Learning Rate**: 2e-5 (AdamW)
- **Epochs**: 10
- **Budget Interval**: $s_{\min} = 0.1$, $s_{\max} = 0.9$
- **Temperature Schedule**: Tau annealed from 2.0 to 0.1
- **Loss Formulation**: Task cross-entropy + adaptive budget penalty loss + annealed entropy regularization

## Citation

If you find BudgetFormer useful in your research, please cite the following submission:

```bibtex
@inproceedings{mbaye2027budgetformer,
  title={BudgetFormer: Adaptive Head Budgeting for Efficient Multi-Head Attention},
  author={Mbaye Abdoulaye,  Faye Bilal, Azzag Hanane and Lebbah Mustapha},
  booktitle={European Symposium on Artificial Neural Networks, Computational Intelligence and Machine Learning (ESANN)},
  year={2027},
  note={Submitted / Under review}
}
```
