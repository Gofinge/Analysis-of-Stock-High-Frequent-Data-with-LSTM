class Evaluator:
    def __init__(self):
        pass

    def evaluate_trend(self, y_true, y_pred):
        size = len(y_true)
        correct = 0
        for i in range(size):
            if y_true[i] * y_pred[i] >= 0:
                correct += 1
        return correct/size

    def evaluate_divided_trend(self, y_true, y_pred, part_num=10):
        size = len(y_true)
        part_size = size // part_num
        acc_list = []
        for i in range(part_num):
            part_y_true = y_true[i*part_size:(i+1)*part_size]
            part_y_pred = y_pred[i*part_size:(i+1)*part_size]
            acc_list.append(self.evaluate_trend(part_y_true, part_y_pred))
        return acc_list

    def evaluate_one_hot_trend(self, y_true, y_pred):
        size = len(y_true)
        correct = 0
        for i in range(size):
            v1, v2 = list(y_true[i]), list(y_pred[i])
            try:
                if v1.index(1) == v2.index(1):
                    correct += 1
            except ValueError:
                print(v1, v2)
        return correct/size

    def evaluate_divided_one_hot_trend(self, y_true, y_pred, part_num=10):
        size = len(y_true)
        part_size = size // part_num
        acc_list = []
        for i in range(part_num):
            part_y_true = y_true[i * part_size:(i + 1) * part_size]
            part_y_pred = y_pred[i * part_size:(i + 1) * part_size]
            acc_list.append(self.evaluate_one_hot_trend(part_y_true, part_y_pred))
        return acc_list