# BudgetFormer: Dynamic Head Budget Allocation for Efficient Transformers

BudgetFormer (AdaTransformer) is an adaptive framework that dynamically allocates the number of active attention heads per input sample. By learning a sample-dependent computational budget rather than relying on a static, over-parameterized multi-head configuration, BudgetFormer matches or surpasses standard Transformer accuracy while significantly reducing computational complexity (FLOPs), activation memory, and carbon footprint.

## Key Features

- **Dynamic Head Budget Allocation**: Predicts an adaptive computation budget per sample using a lightweight gating network and budget predictor, activating only the most relevant attention heads.
- **Over-Parameterization Mitigation**: Eliminates redundant attention heads dynamically, showing that optimal performance is frequently achieved with far fewer active heads than standard configurations.
- **Broad Model Support**: Implemented for both from-scratch encoders/decoders and standard pre-trained backbones (BERT, RoBERTa, DistilBERT, DeBERTa, Longformer).
- **Efficiency and Green AI Profiling**: Built-in estimators for theoretical FLOPs, activation memory, parameter memory, and carbon emissions (gCO2).
- **Comprehensive Ablation Modes**: Evaluates dynamic allocation against fixed budget oracles, random top-k selection, and uniform baselines.

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

- **AG News**: [ag_news](https://huggingface.co/datasets/ag_news) - Topic classification (4 classes)
- **IMDb**: [imdb](https://huggingface.co/datasets/imdb) - Binary sentiment classification
- **DBpedia-14**: [dbpedia_14](https://huggingface.co/datasets/dbpedia_14) - Large-scale topic categorization (14 classes)
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

## Attention Allocation Methods

- **Dynamic Budget Allocation**: Input-dependent budget predictor combined with temperature-annealed head importance scores.
- **Fixed Budget Constraints**: Fixed budget factor ($s \in \{0.2, 0.4, 0.6, 0.8\}$) to evaluate performance at controlled sparsity levels.
- **Top-k Random Selection**: Retains active budget size while picking heads randomly to assess the value of learned routing.
- **Softmax Gating**: Soft weighting of all heads versus hard top-k routing with sparse execution.

## Experimental Results

Empirical results across NLP benchmarks show that dynamic budget allocation matches or exceeds the dense baseline while reducing computation:

| Dataset | Dense Baseline | BudgetFormer (Dynamic) | Fixed Budget (s=0.2) |
| :--- | :--- | :--- | :--- |
| AG News | 91.61% | 92.14% | 91.58% |
| IMDb | 74.35% | 74.98% | 74.64% |
| SNLI | 80.38% | 81.26% | 81.30% |
| DBpedia-14 | 98.49% | 98.56% | - |
| Yelp Review Full | 54.96% | 55.76% | - |

Key findings:
- Increasing attention heads from 1 to 8 or 12 provides diminishing returns on standard benchmarks.
- Dynamic allocation automatically identifies the minimal required heads per task without manual hyperparameter search.
- Significant reduction in FLOPs, activation memory, and carbon footprint during inference.

## Interactive Notebook

An end-to-end interactive demo containing model definition, training loops, profiling (FLOPs, memory, carbon), and attention head visualization heatmaps is available at:
- `AdaTransformer/AdaTransformer.ipynb`

## Default Training Configuration

- **Batch Size**: 16
- **Learning Rate**: 2e-5 (AdamW)
- **Optimizer**: AdamW
- **Epochs**: 10
- **Model Dimension (d_model)**: 768
- **Number of Heads**: 8 (or 12 for larger backbones)
- **Budget Interval**: s_min = 0.1, s_max = 0.9
- **Temperature Schedule**: Tau annealed from 2.0 to 0.1
- **Loss Formulation**: Task cross-entropy + adaptive budget penalty loss
