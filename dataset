# Attentive Feature Mixup (AFM)

This repository contains the code for the paper:

Xiaojiang Peng*, Kai Wang*, Zhaoyang Zeng*, Qing Li, Jianfei Yang, and Yu Qiao, "Suppressing Mislabeled Data via Grouping and Self-Attention", ECCV2020 (* equal contribution).

Paper link: [https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123610766.pdf](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123610766.pdf)

## Abstract
Deep networks achieve excellent results on large-scale clean data but degrade significantly when learning from noisy labels. To suppressing the impact of mislabeled data, this paper proposes a conceptually simple yet efficient training block, termed as Attentive Feature Mixup (AFM), which allows paying more attention to clean samples and less to mislabeled ones via sample interactions in small groups. Specifically, this plug-and-play AFM first leverages a group-to-attend module to construct groups and assign attention weights for group-wise samples, and then uses a mixup module with the attention weights to interpolate massive noisy-suppressed samples. The AFM has several appealing benefits for noise-robust deep learning. (i) It does not rely on any assumptions and extra clean subset. (ii) With massive interpolations, the ratio of useless samples is reduced dramatically compared to the original noisy ratio. (iii) It jointly optimizes the interpolation weights with classifiers, suppressing the influence of mislabeled data via low attention weights. (iv) It partially inherits the vicinal risk minimization of mixup to alleviate over-fitting while improves it by sampling fewer feature-target vectors around mislabeled data from the mixup vicinal distribution. Extensive experiments demonstrate that AFM yields state-of-the-art results on two challenging real-world noisy datasets: Food101N and Clothing1M.

![Figure1](imgs/Figure1.png)


Figure 1: Suppressing mislabeled samples by grouping and self-attention mixup. Different colors and shapes denote given labels and ground truths. Thick and
thin lines denote high and low attention weights, respectively. 

![Figure2](imgs/Figure2.png)

Figure 2: The pipeline of Attentive Feature Mixup (AFM). 


## Requirements

* Linux OS
* Python3.7

## Getting Started

* Install packages `torch` and `torchvision`
```bash
pip install torch
pip install torchvision
```

* Clone this repo:
```bash
git clone https://github.com/kaiwang960112/AFM
cd AFM
```

* Download `Food101` and `Food101N` datasets
```bash
mkdir -p data/food
cd data/food
wget https://food101n.blob.core.windows.net/food101n/Food-101N_release.zip
unzip Food-101N_release.zip
wget http://data.vision.ee.ethz.ch/cvl/food-101.tar.gz
tar xf food-101.tar.gz
```

* Train the model
```bash
python train.py
```
