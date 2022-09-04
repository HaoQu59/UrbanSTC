# UrbanSTC

In this study, we use contrastive self-supervision learning for the fine-grained urban flow inference.

*This is an easy implement of UrbanSTC using Pytorch 1.6, tested on Ubuntu 16.04 with a RTX 2080 GPU.*

# Paper

Hao Qu, Yongshun Gong∗, Meng Chen, Junbo Zhang, Yu Zheng, and Yilong Yin. "[Forecasting Fine-grained Urban Flows via Spatio-temporal Contrastive Self-Supervision](https://www.computer.org/csdl/journal/tk/5555/01/09864246/1G2VMmbOYtG)", TKDE 2022.

If you find our code and dataset useful for your research, please cite our paper:

```
@article{qu2022forecasting,
  title={Forecasting Fine-Grained Urban Flows Via Spatio-Temporal Contrastive Self-Supervision},
  author={Qu, Hao and Gong, Yongshun and Chen, Meng and Zhang, Junbo and Zheng, Yu and Yin, Yilong},
  journal={IEEE Transactions on Knowledge \& Data Engineering},
  number={01},
  pages={1--17},
  year={2022},
  publisher={IEEE Computer Society}
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

