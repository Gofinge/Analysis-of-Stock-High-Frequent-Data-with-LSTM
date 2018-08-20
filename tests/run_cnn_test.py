from HF.config import *

lstm_conf = Config()
lstm_conf.update(feature_name=['buy5', 'bc5', 'buy4', 'bc4', 'buy3', 'bc3', 'buy2', 'bc2', 'buy1', 'bc1',
                               'sale1', 'sc1', 'sale2', 'sc2', 'sale3', ''])