# Analysis-of-Stock-High-Frequent-Data-with-ML

Introduction
====
This project aims at predicting stock price based on high frequency stock data. There is a big difference between
high frequency data and others, thus certain preprocessing methods are necessary in mining useful information.
LSTM is again proved effective in this problem. As a contrast, we also tested some other classical machine learning model such as
XGBoost and random forest.

Experiment
====
Prediction of next tick's price:
-------
We use LSTM to predict stock price, mid-price of next tick. Random Forest and XGBoost are used to classify the following price trend.
- label: *next price delta*
<p class="half" align="center">
  <img src="https://github.com/Gofinge/Analysis-of-Stock-High-Frequent-Data-with-ML/blob/master/plot/rg_lstm_npd.png"/>
  <img src="https://github.com/Gofinge/Analysis-of-Stock-High-Frequent-Data-with-ML/blob/master/plot/cl_rf.png"/>
</p>

- label: *next mid price delta*
<p class="half" aligh="center">
  <img src="https://github.com/Gofinge/Analysis-of-Stock-High-Frequent-Data-with-ML/blob/master/plot/rg_lstm_mpd.png"/>  
</p>

Prediction of future mean price:
-------
- label: *2.5 min mean price delta*
<p class="half" aligh="center">
  <img src="https://github.com/Gofinge/Analysis-of-Stock-High-Frequent-Data-with-ML/blob/master/plot/rg_lstm_mean.png"/>  
</p>

Feature importance:
-------
The size of circle indicates its feature importance.
- model: Random Forest, label: *next price delta*

<p class="half" aligh="center">
  <img src="https://github.com/Gofinge/Analysis-of-Stock-High-Frequent-Data-with-ML/blob/master/plot/fi_rf_npd.png"/>  
</p>

- model: XGBoost, label: *next price delta*
<p>
  <img src="https://github.com/Gofinge/Analysis-of-Stock-High-Frequent-Data-with-ML/blob/master/plot/fi_xgboost_npd.png"/>  
</p>
