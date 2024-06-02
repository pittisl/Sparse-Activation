# Achieving Sparse Activation in Small Language Models

## Introduction
This is the official code repository for the paper ["Achieving Sparse Activation in Small Language Models"](). We aim to achieve sparse activation in SLMs. We demonstrated that the existing magnitude-based sparse activation cannot be applied to SLMs, and using gradient-based attribution scores for sparse activation is a better choice. By applying a corrective term onto the existing GxO attribution metric, our approach can achieve 80% sparsification ratio on SLMs with <5% accuracy loss.

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

### Apply sparse activation based on output magnitude and different attribution scores and plot the accuracy-sparsity trade-off curves
#### magnitude
```
python3 main.py --metric_name magnitude
```
#### gradient
```
python3 main.py --metric_name gradient
```
#### GxO
```
python3 main.py --metric_name gxo
```
#### SNIP
```
python3 main.py --metric_name snip
```
#### Corrected GxO
```
python3 main.py --metric_name cor_gxo
```
We can then check the accuracy-sparsity trade-off curves in the path: `result/truthfulqa/res/both`.

## Citation
```

```
