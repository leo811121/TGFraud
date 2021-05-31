# TGFraud
![image](https://github.com/leo811121/TGFraud/blob/master/pictures/model.PNG)

Currently, most research on fraud detection of online platforms is strive for applying reviews or rating networks on
identifying fraudsters. However, due to the rising importance of personal privacy, rating or review information may be unavailable for
researcher to analyze user’s veracity. Therefore, we propose TGFRAUD, an end-to-end model based on GraphSage and self-attention
mechanism to capture regular pattern for fraudsters on temporal networks. Experimental results on three real-world datasets indicates
that TGFRAUD performs same or even better when competing with other rating-based models no matter on transductive or inductive
settings. Meanwhile, TGFRAUD gives sufficient explainability on the decision of fraud identification via inspecting rating burstiness in a
series of rating history.
