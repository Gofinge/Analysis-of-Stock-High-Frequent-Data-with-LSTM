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
<p class="half" align="center">
  <img src="https://github.com/Gofinge/Analysis-of-Stock-High-Frequent-Data-with-ML/edit/master/plot/rg_lstm_npd.png" width="350px" height="350px"/>
  <img src="https://github.com/Gofinge/Analysis-of-Stock-High-Frequent-Data-with-ML/edit/master/plot/rg_lstm_mpd.png" width="350px" height="350px"/>
  <img src="https://github.com/Gofinge/Analysis-of-Stock-High-Frequent-Data-with-ML/edit/master/plot/cl_rf_npd.png" width="350px" height="350px"/>
</p>
Prediction of future mean price:
-------
