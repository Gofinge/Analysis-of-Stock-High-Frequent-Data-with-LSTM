from HF.config import *
from HF.network import *
from HF.utils import *
from HF.evaluator import Evaluator
from keras import backend as K
import matplotlib.pyplot as plt
import warnings
import os
import pandas as pd
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings("ignore")
K.clear_session()


conf = Config()

# step 1: Get dataset (csv)
data = pd.read_csv('data.csv', encoding='gbk')

# step 2: Select Feature
price = data['price']
average_price = data['VM_Avg_price']
mid_price = data['mid_price']

plt.figure(figsize=(200, 15))
plt.plot(price)
plt.plot(average_price)
plt.plot(mid_price)
plt.show()