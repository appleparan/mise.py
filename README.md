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
* Run in Singularity (>= 3.6) container generated `torch-py3.def`
* Run singularity with following commands
    - bind paths (`/input` for input, `/mnt/data` for output)
    - enable nvidia devices by `--nv`
    - containers refer `requirements.txt` generated from following command
    ```
    $ poetry export -f requirements.txt --output requirements.txt --without-hashes
    ```
* Running models in container (singularity) with commands
    ```
    TORCH_IMG=my_torch_img.sif
    CASES=rnn_mul_lstnet_skip_mccr
    CASE_NAME=210818_LSTNet_MCCR
    mkdir -p /data/appleparan/"${CASE_NAME}"
    singularity exec --nv --bind "${HOME}"/input:/input:ro,/data/"${CASE_NAME}":/mnt/data:rw ${TORCH_IMG} python3 -m mise --dl ${CASES}
    ```

# Reference
* LSTNet: [arxiv link](https://arxiv.org/abs/1703.07015)
    - Lai, Guokun, et al. "Modeling long-and short-term temporal patterns with deep neural networks." The 41st International ACM SIGIR Conference on Research & Development in Information Retrieval. 2018.
* Transformer: [arxiv link](https://arxiv.org/abs/2010.02803)
    - Zerveas, George, et al. "A Transformer-based Framework for Multivariate Time Series Representation Learning." arXiv preprint arXiv:2010.02803 (2020).
* MCCR (Correntropy based Loss): [JLMR link](https://www.jmlr.org/papers/volume16/feng15a/feng15a.pdf)
    - Feng, Yunlong, et al. "Learning with the maximum correntropy criterion induced losses for regression." J. Mach. Learn. Res. 16.1 (2015): 993-1034.

# Publication
* Jongsu Kim and Changhoon Lee. "Deep Particulate Matter Forecasting Model Using Correntropy-Induced Loss." arXiv preprint arXiv:2106.03032 (2021). [link](https://arxiv.org/abs/2106.03032)
    - Accepted in [Journal of Mechanical Science and Technology](https://www.springer.com/journal/12206)
    

