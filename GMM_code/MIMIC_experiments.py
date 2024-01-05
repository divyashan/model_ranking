from GMM import GMM
import sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import argparse
import pickle

from sklearn.model_selection import train_test_split

sys.path.append("/data/ddmg/frank/divya/model_ranking/")
from ranking_helpers import get_model_values_df


def load_test_train(dataset, outcome, model_alg_combinations, split, save_path):
    expt_config = {
            "dataset": dataset + '_' + outcome, # eg mimic+hospitalization
            "model_alg_combinations": tuple(model_alg_combinations),
            "split": split,
        }
    
    predictions = get_model_values_df(expt_config, 'labeled', value='prob_prediction')
    predictions_cleaned = predictions.copy()
    predictions_cleaned['patient_id'] = predictions_cleaned['metadata'].apply(lambda x: x[-1])
    predictions_cleaned.drop('metadata', axis=1, inplace=True)

    grouped = predictions_cleaned.groupby('patient_id')
    patient_dfs = [group for _, group in grouped]
    train_dfs, test_dfs = train_test_split(patient_dfs, test_size=0.5, random_state=42)
    train_df = pd.concat(train_dfs).reset_index(drop=True)
    test_df = pd.concat(test_dfs).reset_index(drop=True)

    save_path_train = save_path + "/" + "datasets/" + f"{dataset}_{outcome}_train.csv"
    save_path_test = save_path + "/" + "datasets/" + f"{dataset}_{outcome}_test.csv"
    train_df.to_csv(save_path_train, index=False)
    test_df.to_csv(save_path_test, index=False)

    return train_df, test_df, expt_config


def train_GMM(train_df, expt_config, n_models, n_labeled, n_classes=2, labeled_data_weight=5):
    np.random.seed(42)
    preds_only = train_df[list(expt_config['model_alg_combinations'])].to_numpy()

    preds_only[preds_only == 0] = 1e-4
    preds_only[preds_only == 1] = 1-1e-4
    transformed_preds = np.log(preds_only/(1-preds_only))
    labels_only = train_df['label'].to_numpy()

    observed_idxs = np.random.choice(len(labels_only), n_labeled, replace=False)
    
    print("Labeled data weight: ", labeled_data_weight)
    print("Number of labeled data points: ", n_labeled)

    gmm_local = GMM(k=n_classes, dim=3, labeled_data_weight=labeled_data_weight)
    X = transformed_preds[:,:n_models]
    observed_labels = labels_only[observed_idxs].astype(int)

    gmm_local.init_em(X, observed_idxs=observed_idxs, observed_labels=observed_labels)
    gmm_local.init_params_with_kmeans(X, initialization="clusters")
    gmm_local.fit(num_iters=1, tol=1e-4)

    estimated_params_w_labeled = gmm_local.get_params()
    return estimated_params_w_labeled


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process arguments.')
    parser.add_argument('--dataset', type=str, default='mimic', help='dataset name')
    parser.add_argument('--outcome', type=str, default='hospitalization', help='outcome name (e.g. hospitalization)')
    # parser.add_argument('--model_alg_combinations', type=str, default='logistic_regression', help='model_alg_combinations')
    parser.add_argument('--n_models', type=int, default=5, help='number of models')
    parser.add_argument('--n_labeled', type=int, default=100, help='number of labeled data points')
    parser.add_argument('--labeled_data_weight', type=float, default=10, help='weight of labeled data points')
    parser.add_argument('--save_path', type=str, default='/data/ddmg/frank/results/mimic', help='save path')

    args = parser.parse_args()
    dataset = args.dataset
    outcome = args.outcome
    n_models = args.n_models
    assert(n_models == 5 or n_models == 3)
    n_labeled = args.n_labeled
    labeled_data_weight = args.labeled_data_weight
    assert(labeled_data_weight > 0)
    save_path = args.save_path

    # print("loaded arguments")

    model_alg_combinations = ('LR_LBFGS', 'DecisionTree_RandomForest', 
                                       'MLP_ERM', 'CART_Heuristic', 'NEWS_Heuristic')
    split = 'labeled'

    train_df, test_df, expt_config = load_test_train(dataset, outcome, model_alg_combinations, split, save_path)

    estimated_params_w_labeled = train_GMM(train_df, expt_config, n_models, n_labeled, n_classes=2, labeled_data_weight=labeled_data_weight)
    # print(estimated_params_w_labeled)
    save_path_params = save_path + "/" + "parameters_trained/" + f"{dataset}_{outcome}_n_labeled_{n_labeled}_n_models_{n_models}_weight_{labeled_data_weight}_params.pkl"
    with open(save_path_params, 'wb') as f:
        pickle.dump(estimated_params_w_labeled, f)
    print(f"Estimated parameters saved to {save_path_params}")







