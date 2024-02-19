import config
from .Conv_models import *
from .FC_models import *
from .LSTM_models import *


def setup():
    if config.MODEL_TYPE == 'fc':
        model = CoreFC(config.CODED_SIZE, config.PATCH_SIZE)
        config.RESIDUAL = False
    elif config.MODEL_TYPE == 'fc_res':
        model = Residual2CoreFC(config.CODED_SIZE, config.PATCH_SIZE, config.REPEAT)
        config.RESIDUAL = True
    elif config.MODEL_TYPE == 'conv':
        model = ConvolutionalCore(config.CODED_SIZE, config.PATCH_SIZE)
        config.RESIDUAL = False
    elif config.MODEL_TYPE == 'conv_res':
        model = ResidualConvolutional(config.CODED_SIZE, config.PATCH_SIZE, config.REPEAT)
        config.RESIDUAL = True
    elif config.MODEL_TYPE == 'lstm':
        model = LSTMCore(config.CODED_SIZE, config.PATCH_SIZE, config.BATCH_SIZE, config.REPEAT)
        config.RESIDUAL = False
    elif config.MODEL_TYPE == 'lstm_res':
        model = ResidualLSTM(config.CODED_SIZE, config.PATCH_SIZE, config.BATCH_SIZE, config.REPEAT)
        config.RESIDUAL = True
    elif config.MODEL_TYPE == 'lstm_mix':
        model = LSTMMix(config.CODED_SIZE, config.PATCH_SIZE, config.BATCH_SIZE, config.REPEAT)
        config.RESIDUAL = None
    else:
        raise Exception("Not implemented")
    return model
