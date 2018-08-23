import numpy as np


class StackingLayer:
    def __init__(self, k_fold=5, *model_list):
        self._k_fold = k_fold
        self._model_list = model_list

    def train(self, train_x, train_y, test_x):
        each_size = int(len(train_x) / self._k_fold)
        train_x_list = []
        train_y_list = []
        for i in range(self._k_fold):
            train_x_list.append(np.array(train_x)[i * each_size:(i + 1) * each_size])
            train_y_list.append(np.array(train_y)[i * each_size:(i + 1) * each_size])

        train_predict_list = []
        each_test_predict_list = []
        test_predict_list = []
        for model in self._model_list:
            for i in range(self._k_fold):
                each_train_x, each_train_y, each_test_x, _ = self.get_k_fold_train_and_test(train_x_list,
                                                                                            train_y_list, i)
                model.fit(each_train_x, each_train_y)
                train_predict_list.append(model.predict(each_test_x))
                each_test_predict_list.append(np.array(model.predict(test_x)))
            test_predict_list.append(self.get_average(each_test_predict_list))
        return test_predict_list

    def get_k_fold_train_and_test(self, train_x_list, train_y_list, index):
        train_x = []
        train_y = []
        test_x = train_x_list[index]
        test_y = train_y_list[index]
        for i in range(self._k_fold):
            if i == index:
                continue
            train_x.extend(train_x_list[i])
            train_y.extend(train_y_list[i])

        return train_x, train_y, test_x, test_y

    def get_average(self, pred_list):
        average = pred_list[0]
        for i in range(1, self._k_fold):
            for j in range(len(average)):
                average[j] += pred_list[i][j]
        for j in range(len(average)):
            average[j] /= self._k_fold
        return average
