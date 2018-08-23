import os

eps = 0.005
z_95 = 1


class Config(dict):
    def __init__(self):
        # name of features
        self['feature_name'] = ['buy1', 'bc1', 'sale1', 'sc1']

        # name of labels
        self['label_name'] = ['30s_delta']

        # number of features
        self['feature_num'] = len(self['feature_name'])

        # proportion of training set: training / (training + testing)
        self['training_set_proportion'] = 0.5

        # time step (each step takes 3 sec)
        self['time_step'] = 5

        # batch size (number of window)
        self['batch_size'] = 150

        # epoch
        self['epoch'] = 100

        self.path = os.getcwd()[:-5]
        # data file path
        self['data_file_path'] = self.path + '/data/SH600031_18.6.15-18.6.20.csv'

        # use previous model_weight
        self['use_previous_model'] = False

        # model_weight file path
        self['model_file_path'] = self.path + 'model_weight/'

        # save file name
        self['save_file_name'] = 'test' + '.h5'

        # load file name
        self['load_file_name'] = 'test' + '.h5'

    def update(self, **kwargs):
        for key in kwargs:
            if key == 'feature_name':
                self['feature_num'] = len(kwargs[key])
            self[key] = kwargs[key]


class LSTM_Config(Config):
    def __init__(self):
        Config.__init__(self)
        # name of features
        self['feature_name'] = ['buy1', 'bc1', 'sale1', 'sc1', 'mid_price', 'wb', 'amount', 'price']

        # name of labels
        self['label_name'] = ['next_delta']

        # number of features
        self['feature_num'] = len(self['feature_name'])

        # time step (each step takes 3 sec)
        self['time_step'] = 5

        # batch size (number of window)
        self['batch_size'] = 150

        # number of neurons of each LSTM layer
        self['LSTM_neuron_num'] = [20, 10, 5]

        # epoch
        self['epoch'] = 100

        # save file name
        self['save_file_name'] = 'lstm_test' + '.h5'

        # load file name
        self['load_file_name'] = 'lstm_test' + '.h5'


class CNN_Config(Config):
    def __init__(self):
        Config.__init__(self)
        # name of features
        self['feature_name'] = ['buy5', 'bc5', 'buy4', 'bc4', 'buy3', 'bc3', 'buy2', 'bc2', 'buy1', 'bc1',
                                'sale1', 'sc1', 'sale2', 'sc2', 'sale3', 'sc3', 'sale4', 'sc4', 'sale5', 'sc5']

        # name of labels
        self['label_name'] = ['mid_price_delta']

        # number of features
        self['feature_num'] = len(self['feature_name'])

        # time step (each step takes 3 sec)
        self['time_step'] = 10

        # batch size (number of window)
        self['batch_size'] = 150

        # epoch
        self['epoch'] = 100

        # save file name
        self['save_file_name'] = 'cnn_test' + '.h5'

        # load file name
        self['load_file_name'] = 'cnn_test' + '.h5'

        # file name
        self['file_name'] = 'cnn_test.h5'


class LM_Config(Config):
    def __init__(self):
        Config.__init__(self)

        # name of features
        self['feature_name'] = ['price', 'mid_price', 'VW_Avg_buy_price', 'VW_Avg_sale_price', 'aggressor_side',
                                'relative_buy_vol', 'relative_sale_vol', 'VW_Avg_price',
                                'VW_Avg_price_minus_current_price', 'vol', 'amount', 'cjbs', 'yclose', 'syl1', 'syl2',
                                'buy1', 'buy2', 'buy3', 'buy4', 'buy5', 'sale1', 'sale2', 'sale3', 'sale4', 'sale5',
                                'bc1', 'bc2', 'bc3', 'bc4', 'bc5', 'sc1', 'sc2', 'sc3', 'sc4', 'sc5', 'wb', 'lb', 'zmm',
                                'buy_vol', 'buy_amount', 'sale_vol', 'sale_amount', 'w_buy', 'w_sale', 'bc1_minus_sc1',
                                'bc2_minus_sc2', 'bc3_minus_sc3',
                                'MACD_DIF', 'MACD_DEA', 'MACD_hist', 'RSI_3', 'RSI_6', 'RSI_12', 'BOLL_upper',
                                'BOLL_middle', 'BOLL_lower']

        # self['feature_name'] = ['bc1', 'sc1', 'buy1', 'sale1']

        # name of labels
        self['label_name'] = ['30s_mean_price_delta']

        # number of features
        self['feature_num'] = len(self['feature_name'])

        # file name
        self['file_name'] = 'lm_test.h5'
