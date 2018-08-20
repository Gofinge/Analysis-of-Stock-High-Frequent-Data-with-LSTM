from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Flatten, Activation, Reshape
from keras.layers import LSTM, Conv2D, BatchNormalization
from keras.optimizers import SGD
from utils import *


class Network:
    def __init__(self, conf):
        self._window_num = conf['window_num']
        self._time_step = conf['time_step']
        self._feature_num = conf['feature_num']
        self._LSTM_neuron_num = conf['LSTM_neuron_num']
        self._LSTM_layer_num = len(self._LSTM_neuron_num)
        self._batch_size = conf['batch_size']
        self._epoch = conf['epoch']
        self._shape = (conf['time_step'], conf['feature_num'])
        self._model_file_path = conf['model_file_path']
        self._model = self._init_model()
        self.print_model_summary()

    def _init_model(self):
        model = Sequential()
        input_shape = (self._time_step, self._feature_num)
        kernel_size = (3, 3)
        filters_num = 4

        # model.add(Conv2D(input_shape=input_shape, filters=filters_num, kernel_size=kernel_size, strides=(1, 1)))
        # model.add(BatchNormalization())
        # model.add(Activation('relu'))

        # model.add(Reshape((1, (self._time_step-kernel_size[0]+1)*(self._feature_num-kernel_size[1]+1)*filters_num)))

        model.add(LSTM(input_shape=input_shape, units=20, return_sequences=True))

        model.add(Flatten())
        # model.add(Dense(units=3, activation='softmax'))
        model.add(Dense(units=1, activation='linear'))

        # sgd = SGD(lr=0.1, momentum=0.9, decay=0.1)
        model.compile(loss=my_loss, optimizer='adam', metrics=['acc'])
        return model

    def save(self, file_name):
        self._model.save_weights(self._model_file_path + file_name)

    def load(self, file_name):
        self._model.load_weights(self._model_file_path + file_name)

    def train(self, train_x, train_y):
        self._model.fit(train_x, train_y, batch_size=self._batch_size, epochs=self._epoch, verbose=1)

    # TODO: strong train
    def strong_train(self):
        pass

    def predict(self, x):
        # y_prob = self._model.predict(x, batch_size=self._batch_size)
        # y_label = []
        # for vector in y_prob:
            # y_label.append(labelize_prob_vector(vector))
        # return y_label
        return self._model.predict(x, batch_size=1)

    def dynamic_correct(self):
        pass

    def print_model_summary(self):
        print(self._model.summary())