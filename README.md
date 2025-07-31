# KG‑BiLM: Knowledge Graph Embedding via Bidirectional Language Models

KG‑BiLM is a **three‑step recipe** for turning a decoder‑style large language model into a powerful knowledge‑graph encoder:

1. **Bidirectional Knowledge Attention (BKA)** – we lift the causal mask so every token/entity can attend to both past *and* future neighbours that are within a prescribed hop distance in the graph.
2. **Knowledge‑Masked Prediction (KMP)** – a structure‑aware variant of masked LM that hides entity/text tokens and asks the model to recover them from the *preceding* representation, forcing it to leverage global KG cues as well as local context.
3. **Contrastive Graph Semantic Aggregation (CGSA)** – an InfoNCE loss that pulls together two augmented views of the same sub‑graph and pushes away others, preserving discriminative topology. fileciteturn1file9L86-L90

> The combination yields unified embeddings that respect KG structure *and* nuanced linguistic semantics, surpassing prior KGE and PLM baselines on WN18RR, FB15k‑237, Wikidata5M, and zero‑shot entity splits.

------

## Installation

```bash
pip install kg-bilm            # lightweight wrapper
pip install flash-attn --no-build-isolation  # (optional) faster attention
```

You can also install from source for bleeding‑edge updates:

```bash
git clone https://github.com/your‑org/kg-bilm.git
pip install -e kg-bilm
```

------

## Getting Started

```python
import torch
from kgbilm import KGBiLM

model = KGBiLM.from_pretrained(
    "czr/kg-bilm-bka-mntp",   # base checkpoint after BKA+KMP
    peft_model_name_or_path="zirui-chen/kg-bilm-bka-mntp-cgsa",  # optional CGSA LoRA
    device_map="auto",
    torch_dtype=torch.bfloat16,
)

triples = [
    ("Beethoven", "genre", "Classical music"),
    ("Washington Wizards", "playsInLeague", "NBA"),
]
embeddings = model.encode(triples)  # (batch, dim)
```

The encoder returns one vector per triple, which can be pooled or fed into downstream link‑prediction heads.

------

## Model Zoo

| Base LLM    | BKA + KMP | + CGSA (unsup) | HuggingFace                    |
| ----------- | --------- | -------------- | ------------------------------ |
| Qwen-2.5‑7B | ✅         | ✅              | `czr/KG‑BiLM‑Qwen‑2.5‑7B‑cgsa` |

------

## Training

### KMP (analogous to MNTP)

Run the *knowledge‑masked prediction* stage with a single JSON config:

```bash
python experiments/run_kmp.py train_configs/kmp/qwen2.5.json
```

The config mirrors HuggingFace MLM configs with graph‑specific additions:

```json
{
  "model_name_or_path": "Qwen/Qwen2.5-7B-Instruct",
  "kg_dataset": "wikidata5m",
  "mask_ratio": 0.2,
  "hop_threshold": 2,
  "lora_r": 16,
  "gradient_checkpointing": true,
  "torch_dtype": "bfloat16"
}
```

Hyper‑parameters follow the paper default (28 layers, *d*=1024, 200 k steps, lr 1e‑4

### Unsupervised contrastive training (CGSA)

Download the Wikipedia‑1M sentence corpus and start CGSA:

```bash
wget https://huggingface.co/datasets/princeton-nlp/datasets-for-simcse/resolve/main/wiki1m_for_simcse.txt -P cache/
python experiments/run_cgsa.py train_configs/cgsa/qwen2.5.json
```

The script generates two stochastic graph‑aware views per batch and optimises the InfoNCE objective

------

## Evaluation

We provide a plug‑in evaluator for **link prediction** and **triple classification**. For example, to evaluate on WN18RR:

```bash
python experiments/kg_eval.py \
   --model_name zirui-chen/KG-BiLM-Llama3-cgsa \
   --dataset WN18RR
```

KG‑BiLM sets a new SOTA MRR of **0.682** on WN18RR, edging out CP‑KGC and achieves **0.748** MRR in the zero‑shot split of Wikidata5M

------

## Results Snapshot

| Dataset                | Task       | Metric  | KG‑BiLM   | Previous best     |
| ---------------------- | ---------- | ------- | --------- | ----------------- |
| WN18RR                 | Link Pred. | MRR     | **0.682** | 0.673 (CP‑KGC)    |
| FB15k‑237N             | Link Pred. | Hits@10 | **0.546** | 0.538 (CSProm‑KG) |
| Wikidata5M (zero‑shot) | Link Pred. | MRR     | **0.748** | 0.714 (SimKGC)    |

------

## Bugs or Questions?

Open an issue on **GitHub**, or email `kg‑bilm‑maintainers@your‑org.com`.
