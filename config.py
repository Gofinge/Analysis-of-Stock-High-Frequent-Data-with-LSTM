class Config(dict):
    def __init__(self):
        # name of features
        self['feature_name'] = ['mid_price', 'buy1', 'buy2', 'bc1', 'bc2', 'sale1', 'sale2']

        # name of labels
        self['label_name'] = ['next_delta']

        # number of features
        self['feature_num'] = len(self['feature_name'])

        # proportion of training set: training / (training + testing)
        self['training_set_proportion'] = 0.5

        self['window_num'] = 1

        # time step (each step takes 3 sec)
        self['time_step'] = 10

        # batch size (number of window)
        self['batch_size'] = 100

        # number of neurons of each LSTM layer
        self['LSTM_neuron_num'] = [8, 16, 8]

        # epoch
        self['epoch'] = 5

        # data file path
        self['data_file_path'] = 'data/data.csv'

        # use previous model
        self['use_previous_model'] = False

        # model file path
        self['model_file_path'] = 'model/'

        self['file_name'] = 'test.h5'