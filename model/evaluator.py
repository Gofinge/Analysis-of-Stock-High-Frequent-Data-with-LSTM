from .config import *


class Evaluator:
    def __init__(self):
        self._acc = 3
        pass

    def evaluate_trend_simple(self, y_true, y_pred):
        size = len(y_true)
        correct = 0
        all = 0
        for i in range(size):
            if y_true[i] * y_pred[i] > 0 and y_true[i] != 0:
                correct += 1
                all += 1
            elif y_true[i] != 0:
                all += 1
        return round(correct / all, self._acc)

    def evaluate_trend(self, y_true, y_pred):
        size = len(y_true)
        correct_rise_or_decline = 0
        correct_stay = 0
        stay = 0
        for i in range(size):
            if abs(y_true[i]) < eps:
                stay += 1

        for i in range(size):
            if abs(y_pred[i]) < eps:
                if abs(y_true[i]) < eps:
                    correct_stay += 1
                else:
                    pass  # 预测价格不变，但实际价格变化，预测错误
            else:
                if abs(y_true[i]) < eps:
                    pass  # 预测价格改变，但实际价格不变，预测错误
                else:
                    if y_pred[i] * y_true[i] > 0:
                        correct_rise_or_decline += 1  # 预测价格变化趋势和实际价格变化趋势相同
                    else:
                        pass  # 预测价格变化趋势和实际价格变化趋势相反

        correct = correct_stay + correct_rise_or_decline
        if stay == 0:
            correct_stay = 0
            stay = 1
        return round(correct / size, self._acc), \
               round(correct_stay / stay, self._acc), \
               round(correct_rise_or_decline / (size - stay), self._acc)

    def evaluate_trend_2(self, y_true, y_pred):
        size = len(y_true)
        correct = 0

        for i in range(size):
            if abs(y_pred[i]) < eps:
                if abs(y_true[i]) < eps:
                    correct += 1
            else:
                j = i
                try:
                    while abs(y_true[j]) < eps:
                        j += 1
                    if y_pred[i] * y_true[j] > 0:
                        correct += 1
                except:
                    pass

        return round(correct / size, self._acc)

    def evaluate_trend_without_stay(self, y_true, y_pred):
        size = len(y_true)
        correct = 0
        for i in range(size):
            j = i
            try:
                while abs(y_true[j]) < eps:
                    j += 1
                if y_pred[i] * y_true[j] > 0:
                    correct += 1
            except IndexError:
                pass
        return round(correct / size, self._acc)

    def evaluate_divided_trend(self, y_true, y_pred, part_num=10):
        size = len(y_true)
        part_size = size // part_num
        acc_list = []
        for i in range(part_num):
            part_y_true = y_true[i * part_size:(i + 1) * part_size]
            part_y_pred = y_pred[i * part_size:(i + 1) * part_size]
            acc_list.append(self.evaluate_trend_simple(part_y_true, part_y_pred))
        return acc_list

    def evaluate_one_hot_trend(self, y_true, y_pred):
        size = len(y_true)
        correct = 0
        all = 0
        for i in range(size):
            v1, v2 = list(y_true[i]), list(y_pred[i])
            try:
                if v1.index(1) == v2.index(1):
                    correct += 1
                all += 1
            except ValueError:
                print(v1, v2)
        print(correct, all)
        return round(correct / all, self._acc)

    def evaluate_divided_one_hot_trend(self, y_true, y_pred, part_num=10):
        size = len(y_true)
        part_size = size // part_num
        acc_list = []
        for i in range(part_num):
            part_y_true = y_true[i * part_size:(i + 1) * part_size]
            part_y_pred = y_pred[i * part_size:(i + 1) * part_size]
            acc_list.append(self.evaluate_one_hot_trend(part_y_true, part_y_pred))
        return acc_list

    def evaluate_mean_and_variance(self, y_true, y_pred):
        pred_mean, pred_std = y_pred[0], y_pred[1]
        all = len(pred_mean)
        correct = 0
        for i in range(all):
            if abs(y_true - pred_mean) < pred_std * z_95:
                correct += 1
        return round(correct / all, self._acc)
