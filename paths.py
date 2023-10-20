import hashlib

MODEL_PATH = '/data/ddmg/frank/saved_models/'
DATA_PATH = '/data/ddmg/frank/data/'
LOG_PATH = '/data/ddmg/frank/logs/'
PREDICTIONS_PATH = '/data/ddmg/frank/predictions/'
RESULTS_PATH = '/data/ddmg/frank/results/'

def create_predictions_path(config):
    # Might have to hash this if there are too many keys.
    config_keys = sorted(config.keys())
    config_str = ''
    for key in config_keys:
        config_str += key + '_' + str(config[key]) + '_'
    return PREDICTIONS_PATH + config_str[:-1]

def create_results_path(config):
    config_keys = sorted(config.keys())
    config_str = ''
    for key in config_keys:
        config_str += key + '_' + str(config[key]) + '_'
    return RESULTS_PATH  + '/' + config['dataset'] + '/' + 'label_pct_' + str(config['labeled_pct']) + '/'


def get_config_hash(d):
    """
    Returns a deterministic hash of a dictionary.
    """
    # Convert the dictionary to a string and encode it as bytes
    dict_str = str(d).encode('utf-8')
    
    # Use the SHA-256 hashing algorithm to generate a hash
    hash_obj = hashlib.sha256(dict_str)
    hash_str = hash_obj.hexdigest()
    
    return hash_str

