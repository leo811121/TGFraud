# TGFraud
![image](https://github.com/leo811121/TGFraud/blob/master/pictures/model.PNG)

- **Intro** Currently, most research on fraud detection of online platforms is strive for applying reviews or rating networks on
identifying fraudsters. However, due to the rising importance of personal privacy, rating or review information may be unavailable for
researcher to analyze user’s veracity. Therefore, we propose TGFRAUD, an end-to-end model based on GraphSage and self-attention
mechanism to capture regular pattern for fraudsters on temporal networks. Experimental results on three real-world datasets indicates
that TGFRAUD performs same or even better when competing with other rating-based models no matter on transductive or inductive
settings. Meanwhile, TGFRAUD gives sufficient explainability on the decision of fraud identification via inspecting rating burstiness in a
series of rating history. 

More details and reseatch about TGFraud can be seen in [here](https://github.com/leo811121/TGFraud/blob/master/TGFraud.pdf).

# result Average AUC Scores of Tenfold Cross-Validation

|            Dataset           | OTC |ALPHA|AMAZON|
| -----------------------------|-----|-----|------|
| REV2 (rating & temporal)     |0.895|0.840|0.854 |
| SGCN (rating & temporal)     |0.957|0.882|0.840 |
| RGCN (rating & temporal)     |0.912|0.839|0.862 |
| GAT (rating & temporal)      |0.912|0.839|0.862 |
| GraphSage(rating & temporal) |0.912|0.839|0.862 |
| TGFraud (temporal only)      |**0.967**|**0.902**|0.738 |

# Code explanation
[models](https://github.com/leo811121/TGFraud/tree/master/models) : TGFraud contains three modules. EdgeSage is for
aggregating neighbor and edge level information. Temporal encoder is for capturing burstiness in temporal series rating history. Temporal-edge aggregator generates node
embeddings possessing both neighbor and temporal series information
- **EdgeSage** -> [gnn_model.py](https://github.com/leo811121/TGFraud/blob/master/models/gnn_model.py), [egsage.py](https://github.com/leo811121/TGFraud/blob/master/models/egsage.py)
- **Temporal aggregator** -> [encoder_temporal.py](https://github.com/leo811121/TGFraud/blob/master/models/encoder_temporal.py), [transformer_encoder.py](https://github.com/leo811121/TGFraud/blob/master/models/transformer_encoder.py)
- **Temporal-edge aggregator** -> [encoder_hybrid.py](https://github.com/leo811121/TGFraud/blob/master/models/encoder_hybrid.py)

[network_data](https://github.com/leo811121/TGFraud/tree/master/network_data) This directory contains three dataset(OTC, ALPHA, AMAZON) and data-preprocessing file.

[gnn_training.py](https://github.com/leo811121/TGFraud/blob/master/training/gnn_training.py) This file runs model training and AUC evaluation.



# Run the code
```
!python TGFRAUD/train.py --dataset alpha
```
