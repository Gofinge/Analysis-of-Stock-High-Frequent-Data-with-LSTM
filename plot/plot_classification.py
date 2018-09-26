import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_classification(y_true, y_pred):
    correct = [y_true[i] ^ y_pred[i] for i in range(len(y_true))]
    height = np.random.random(len(y_true))
    dt = pd.DataFrame(data=list(zip(y_true, y_pred, correct, height)),
                      columns=['true', 'predicted', 'correct', 'height'])
    sns.swarmplot(x='true', y='height', hue='predicted', data=dt)
    plt.ylabel('')
    plt.yticks(range(1), [''])
    plt.show()


if __name__ == '__main__':
    y_true = np.random.randint(0, 2, 50)
    y_pred = np.random.randint(0, 2, 50)
    plot_classification(y_true, y_pred)
