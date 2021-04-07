# Models

## Univariate Statistical Learning Models

1. OU
2. ARIMA (might AR)

## Univariate Machine Learning Models

1. MLP
2. MLP + MCCR Loss
3. Bahdanau Attention
4. Bahdanau Attention + MCCR

### loss function input for Train/Valid/Test set
* Train/Valid : Loss between `y_hat` and `y` (transformed))
* Test : Loss between `y_hat_inv` (inverse transformed) and `y_raw`

## Multivariate Machine Learning Models

1. XGBoost
2. MLP
3. MLP + MCCR
4. LSTNet (Skip Layer)
5. LSTNet (Skip Layer) + MCCR
6. Transformer
7. Transformer + MCCR

### loss function input for Train/Valid/Test set
* Train/Valid : Loss between `y_hat` and `y` (transformed))
* Test : Loss between `y_hat_inv` (inverse transformed) and `y_raw`

# Frameworks
* PyTorch (>= 1.8.0)
* Optuna (>= 2.3.0)
* scikit-learn (>= 0.24.0)

# Container
* Run in Singularity (> 3.x) container generated from [this script](https://github.com/appleparan/SING-scripts/blob/master/torch-py3.def)
* Singualrity excution script: [Link](https://github.com/appleparan/SING-scripts/blob/master/torch)

# Reference
* LSTNet: [arxiv link](https://arxiv.org/abs/1703.07015)
    - Lai, Guokun, et al. "Modeling long-and short-term temporal patterns with deep neural networks." The 41st International ACM SIGIR Conference on Research & Development in Information Retrieval. 2018.
* Transformer: [arxiv link](https://arxiv.org/abs/2010.02803)
    - Zerveas, George, et al. "A Transformer-based Framework for Multivariate Time Series Representation Learning." arXiv preprint arXiv:2010.02803 (2020).
* MCCR (Correntropy based Loss): [JLMR link](https://www.jmlr.org/papers/volume16/feng15a/feng15a.pdf)
    - Feng, Yunlong, et al. "Learning with the maximum correntropy criterion induced losses for regression." J. Mach. Learn. Res. 16.1 (2015): 993-1034.

# Paper
* In Manuscript

