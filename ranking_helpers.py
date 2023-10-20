import numpy as np
import pandas as pd
from paths import create_predictions_path
CAMELYON17_GROUP_VARS = ['WSI_index', 'hospital_index']
CIVILCOMMENTS_GROUP_VARS = ['sex:male', 'sex:female', 'orientation:LGBTQ', 
                            'religion:christian', 'religion:muslim', 'religion:other_religions', 
                            'race:black', 'race:white', 'race:identity_any']


def add_metadata_columns(results_df):
    civilcomments_metadata = ['sex:male', 'sex:female', 'orientation:LGBTQ', 'religion:christian', 'religion:muslim', 'religion:other_religions', 
                              'race:black', 'race:white', 'race:identity_any', 'severe_toxicity', 'obscene', 'threat', 'insult', 'identity_attack', 'sexual_explicit']
            
    for i,metadata_col_name in enumerate(civilcomments_metadata):
        results_df[metadata_col_name] = results_df['metadata'].apply(lambda x: x[i])
    return results_df

def get_model_values_df(config, split, value='accuracy'):
    dataset = config['dataset']
    algs = config['learning_algs']
    model = 'densenet121'
    if dataset == 'civilcomments':
        model='distilbert-base-uncased'

    prob_predictions_matrix = []
    binary_prediction_matrix = []
    accuracy_matrix = []
    metadata_matrix = []

    for alg in algs:
        preds_config = {
            "dataset": dataset,
            "algorithm": alg,
            "split": split,
            "model": model,
        }

        predictions = np.load(create_predictions_path(preds_config) + "/preds.npy")
        metadata = np.load(create_predictions_path(preds_config) + "/metadata.npy")
        
        binarized_preds = np.argmax(predictions, axis=1)
        binary_prediction_matrix.append(binarized_preds)
        prob_predictions_matrix.append(predictions)
        metadata_matrix.append(metadata)

        if split != 'unlabeled':
            labels = np.load(create_predictions_path(preds_config) + "/labels.npy")
            accuracy = binarized_preds == labels
            accuracy_matrix.append(accuracy)

    
    binary_prediction_matrix = np.stack(binary_prediction_matrix, axis=0)   
    prob_predictions_matrix = np.stack(prob_predictions_matrix, axis=0) 
    metadata_matrix = np.stack(metadata_matrix, axis=0)[0]
    if split != 'unlabeled':
        accuracy_matrix = np.stack(accuracy_matrix, axis=0).astype(int)

    #TODO: hacky
    results_df = []
    n_examples = binary_prediction_matrix.shape[1]
    for i in range(n_examples):
        result = {'index': i, 'metadata': metadata_matrix[i]}
        if split != 'unlabeled':
            result['label'] = labels[i]
        for j, alg in enumerate(algs):
            if value == 'accuracy':
                result[alg] = accuracy_matrix[j, i]
            elif value == 'binary_prediction':
                result[alg] = binary_prediction_matrix[j, i]
            elif value == 'prob_prediction':
                result[alg] = prob_predictions_matrix[j, i]
        results_df.append(result)
    
    results_df = pd.DataFrame(results_df)

    # Map metadata to group-labeled columns
    if dataset == 'civilcomments':
        results_df = add_metadata_columns(results_df)
    elif dataset == 'camelyon17':
        results_df['WSI_index'] = results_df['metadata'].apply(lambda x: x[1])
        results_df['hospital_index'] = results_df['metadata'].apply(lambda x: x[0])

    return results_df
