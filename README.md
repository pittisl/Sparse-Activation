# Achieving Sparse Activation in Small Language Models

## Introduction
This is the official code repository for the paper ["Achieving Sparse Activation in Small Language Models"](). Sparse activation selectively activates only an input-dependent set of neurons in inference, is a useful technique to reduce the computing cost of Large Language Models (LLMs) without retraining or adaptation efforts. We proposed a new attribution metric for Small Language Models (SLMs) that can achieve 80% sparsification ratio with $<$5% model accuracy loss.

## Requirement
Install all the required packages.
```
pip install -r requirements.txt
```
## General Usage
We use Phi-2 and the truthfulqa dataset to show an example result of sparse activation. We can use the following steps to generate accuracy-sparsity curves based on different kinds of metrics including the proposed Corrected GxO metric.

### Folder Creation
Create the folder for generated results as following
```
python3 folder_creation.py
```

### Label Generation
Generate the labels for sparse activation
```
python3 label_generation.py
```

### Mganitude Generation
```
python3 Mag_attention.py
```
```
python3 Mag_mlp.py
```

### Attribution scores Generation

#### gradient
```
python3 IG_attention.py --metric gradient
```

#### Gradient*Output (GxO)
```
python3 IG_attention.py --metric gxo
```

#### Integrated gradients (IG) with 20 interpolations (The number of interpolations can be any positive integer)
```
python3 IG_attention.py --metric ig --n_steps 20
```

### Apply sparse activation based on different attribution scores
```
python3 mask_multi_both.py --metric_name magnitude
```
```
python3 mask_multi_both.py --metric_name gradient
```
```
python3 mask_multi_both.py --metric_name gxo
```
```
python3 mask_multi_both.py --metric_name snip
```
```
python3 mask_multi_both.py --metric_name cor_gxo
```
Then we can check the results in the path: "result/truthfulqa/res/both".

## Citation
```

```
