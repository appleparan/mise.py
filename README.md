# Models

## Univariate
1. OU
2. ARIMA (1, 0, 0)
3. ARIMA (1, 0, 1)
4. MLP (direct prediction)
    * Train/Valid : Loss between y_hat and y (raw)
    * Test  : Loss between y_hat and y (raw)
5. MLP + Seasonality (indirect prediction)
    * Train/Valid : Loss between y_hat and y (transformed))
    * Test  : Loss between y_hat_inv (inverse_transformed) and y_raw
6. Sea2Seq (indirect prediction)
7. Bahdanau Attention (indirect prediction)

## Multivariate

1. MLP (direct prediction)
    * Train/Valid : Loss between y_hat and y_raw
    * Test  : Loss between y_hat and y_raw
2. MLP + Seasonality (indirect prediction)
    * Train/Valid : Loss between y_hat and y (transformed)
    * Test  : Loss between y_hat_inv (inverse_transformed) and y_raw
3. LSTNet + Skip Layer (direct prediction)
4. LSTNet + Attention Layer (direct prediction)
5. Temporal Pattern Attention (direct prediction)
6. Transformer + Seasonality (direct prediction)

# Reference
    * LSTNet
    * TPA
    * Transformer

# Paper