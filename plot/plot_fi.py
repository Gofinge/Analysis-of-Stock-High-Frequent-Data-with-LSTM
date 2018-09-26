import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

data = pd.read_csv('pearson.csv')

fi_file = np.load('feature_importance.npy')

feature_num = 20

feature_name = fi_file[0][0:feature_num]
print(feature_name)

feature_importance = fi_file[1][0:feature_num]
feature_importance = list(map(float, feature_importance))
pearson_npd = []
pearson_mpd = []
pearson_2 = []

for name in feature_name:
    p = data[data.feature_name == name][['next_delta', 'mid_price_delta', '2.5min_mean_price_delta']].values[0]
    pearson_npd.append(float(p[0]))
    pearson_mpd.append(float(p[1]))
    pearson_2.append(float(p[2]))

r_list = np.random.random(feature_num)
g_list = np.random.random(feature_num)
colors = [(0.3 + 0.3 * r, 0.5 + 0.4 * g, 0.8) for r, g in zip(r_list, g_list)]

plt.figure(figsize=(10, 10), dpi=300)

for i in range(feature_num):
    plt.scatter(x=i, y=pearson_npd[i], s=1000000*(feature_importance[i]**3)+5, c=colors[i])

scale_list = range(feature_num)
index_list = feature_name
plt.xticks(scale_list, index_list, rotation=90)
plt.ylim((-1, 1))
plt.xlabel('Feature')
plt.ylabel('Pearson')
plt.show()
