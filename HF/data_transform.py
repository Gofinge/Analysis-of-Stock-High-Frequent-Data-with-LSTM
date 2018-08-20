import pandas as pd
import numpy as np


# require that last column is label
def split_feature_label(raw_data, time_step):
    data = np.array(raw_data)
    window_num = data.shape[0] - time_step + 1
    x = []
    y = []
    for i in range(window_num):
        x.append(data[i:time_step + i, 0:data.shape[1] - 1])
        x.append(data[time_step + i - 1, -1])
    return np.array(x), np.array(y)


if __name__ == '__main__':
    raw_data = pd.read_csv('raw.csv', encoding='gb18030')
    output = data_transfrom(raw_data=raw_data, time_step=6)


