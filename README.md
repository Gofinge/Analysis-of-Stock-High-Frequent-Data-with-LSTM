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
label: next price delta
<p class="half" align="center">
  <img src="https://github.com/Gofinge/Analysis-of-Stock-High-Frequent-Data-with-ML/blob/master/plot/rg_lstm_npd.png"/>
  <img src="https://github.com/Gofinge/Analysis-of-Stock-High-Frequent-Data-with-ML/blob/master/plot/cl_rf.png"/>
</p>

label: next mid price delta
<p class="half" aligh="center">
  <img src="https://github.com/Gofinge/Analysis-of-Stock-High-Frequent-Data-with-ML/blob/master/plot/rg_lstm_mpd.png"/>  
</p>

Prediction of future mean price:
-------
