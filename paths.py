MODEL_PATH = '/data/ddmg/frank/saved_models/'
DATA_PATH = '/data/ddmg/frank/data/'
LOG_PATH = '/data/ddmg/frank/logs/'
PREDICTIONS_PATH = '/data/ddmg/frank/predictions/'


def create_predictions_path(config):
    # Might have to hash this if there are too many keys.
    config_keys = sorted(config.keys())
    config_str = ''
    for key in config_keys:
        config_str += key + '_' + str(config[key]) + '_'
    return PREDICTIONS_PATH + config_str[:-1]

