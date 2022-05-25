# UrbanSTC

FUFI is a technique that focuses on inferring fine-grained urban flows depending solely on observed coarse-grained data. However, existing methods always require massive learnable parameters and the complex network structures. To reduce these defects, we formulate a contrastive self-supervision method to predict fine-grained urban flows taking into account all correlated spatial and temporal contrastive patterns.

*This is an easy implement of UrbanSTC using Pytorch 1.6, tested on Ubuntu 16.04 with a RTX 2080 GPU.*

# Dataset

The datasets we use for model training is detailed in Section 5.1.1 of our paper. Here, we release P1 in TaxiBJ (7/1/2013-10/31/2013) for public use. Totally, there 1530, 765 and 765 samples in the training, validation and test set respectively. Besides, the corresponding external factors data (e.g., meteorology, time features) are also included. 

# Model  Training

First, we run `./utils/data_process.py` to create `more coarse-grained flow map` and `TCS dataset`.

Then, train `train_reg_pretrain.py, train_Inference_net.py, train_tc_pretrain.py` to save `encoder` parameters.

Finally, train `train_UrbanSTC.py`

# Model Efficiency

We provide model efficiency in `Test_Model_Efficiency`.
