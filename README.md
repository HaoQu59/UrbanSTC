# UrbanSTC

In this study, we use contrastive self-supervision learning for the fine-grained urban flow inference.

*This is an easy implement of UrbanSTC using Pytorch 1.6, tested on Ubuntu 16.04 with a RTX 2080 GPU.*

# Paper

Hao Qu, Yongshun Gong, Meng Chen, Junbo Zhang, Yu Zheng, and Yilong Yin. "[Forecasting Fine-grained Urban Flows via Spatio-temporal Contrastive Self-Supervision](https://www.computer.org/csdl/journal/tk/5555/01/09864246/1G2VMmbOYtG)", TKDE 2022.

If you find our code and dataset useful for your research, please cite our paper:

```
@ARTICLE {9864246,
author = {H. Qu and Y. Gong and M. Chen and J. Zhang and Y. Zheng and Y. Yin},
journal = {IEEE Transactions on Knowledge &amp; Data Engineering},
title = {Forecasting Fine-Grained Urban Flows Via Spatio-Temporal Contrastive Self-Supervision},
year = {5555},
volume = {},
number = {01},
issn = {1558-2191},
pages = {1-17},
keywords = {task analysis;superresolution;spatial resolution;semantics;neural networks;training;predictive models},
doi = {10.1109/TKDE.2022.3200734},
publisher = {IEEE Computer Society},
address = {Los Alamitos, CA, USA},
month = {aug}
}
```

# Framework

![Image text](https://github.com/HaoQu59/UrbanSTC/blob/main/img/framework.png)

# Result

We evaluate our method on TaxiBJ in four different time periods with different proportions of training data. The main experimental results are shown as follows:

![Image text](https://github.com/HaoQu59/UrbanSTC/blob/main/img/results.png)

![Image text](https://github.com/HaoQu59/UrbanSTC/blob/main/img/results2.png)

![Image text](https://github.com/HaoQu59/UrbanSTC/blob/main/img/results3.png)

![Image text](https://github.com/HaoQu59/UrbanSTC/blob/main/img/results4.png)

# Dataset

Here, we release P1 in TaxiBJ (7/1/2013-10/31/2013) for public use. Totally, there 1530, 765 and 765 samples in the training, validation and test set respectively. Besides, the corresponding external factors data (e.g., meteorology, time features) are also included. 

