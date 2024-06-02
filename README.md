# Achieving Sparse Activation in Small Language Models

## Introduction
This is the official code repository for the paper ["Achieving Sparse Activation in Small Language Models"](). Sparse activation selectively activates only an input-dependent set of neurons in inference, is a useful technique to reduce the computing cost of Large Language Models (LLMs) without retraining or adaptation efforts. We proposed a new attribution metric for Small Language Models (SLMs) that can achieve 80% sparsification ratio with $<$5% model accuracy loss.

## Requirement
Install all the required packages.
```
pip install -r requirements.txt
```
## General Usage
We use Phi-2 and the TruthfulQA dataset to demonstrate an example of sparse activation. The following steps can be used to generate accuracy-sparsity trade-off curves based on various metrics, including the proposed Corrected GxO metric.

### Folder Creation
Create the folder for generated results as following code
```
python3 folder_creation.py
```

### Label Generation
Generate the labels for sparse activation
```
python3 label_generation.py
```

### Mganitude Generation
we can use the following code to generate the output magnitude of each attention head and MLP neurons
```
python3 Mag_attention.py
```
```
python3 Mag_mlp.py
```

### Attribution scores Generation
we can use the following code to generate the various attribution-based scores of each attention head and MLP neurons
#### gradient
```
python3 attribution_attention.py --metric gradient
```

#### Gradient*Output (GxO)
```
python3 attribution_attention.py --metric gxo
```

#### Integrated gradients (IG) with 20 interpolations (The number of interpolations can be any positive integer)
```
python3 attribution_attention.py --metric ig --n_steps 20
```

### Apply sparse activation based on different attribution scores and plot the accuracy-sparsity trade-off curves
```
python3 main.py --metric_name magnitude
```
```
python3 main.py --metric_name gradient
```
```
python3 main.py --metric_name gxo
```
```
python3 main.py --metric_name snip
```
```
python3 main.py --metric_name cor_gxo
```
We can then check the accuracy-sparsity trade-off curves in the path: 'result/truthfulqa/res/both'.

## Citation
```

```
