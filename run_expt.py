import os
import pandas as pd
import numpy as np
import pdb

import swifter
from scipy.stats import kendalltau
from tqdm import tqdm

from paths import create_results_path, get_config_hash
from ranking_models import random_ranking, labeled_data_ranking, dawid_skene_ranking
from ranking_helpers import get_model_values_df


if __name__ == "__main__":
    # All options
    ranking_algs = ['random', 'labeled_global', 'labeled_group_specific', 'oracle', 'dawid_skene', 'nearest_neighbor']
    datasets = ['civilcomments', 'camelyon17']

    # This run's options
    n_runs = 3
    datasets = ['civilcomments']
    list_of_group_vars = [['sex:male',  'race:black', 'religion:muslim'],]
    # list_of_group_vars = [['sex:male',  'race:black'],]

    # datasets = ['camelyon17']
    # list_of_group_vars = [['WSI_index']]


    ranking_algs = ['random', 'labeled_global', 'labeled_group_specific', 'oracle']
    labeled_pcts = [.025, .05, .1, .25, .5, 1.0]

    # Sorting so that the different orderings of group variables produce the same experiment config


    list_of_group_vars = [sorted(x) for x in list_of_group_vars]
    group_definitions = [','.join(x) for x in list_of_group_vars]

    expt_configs = []
    for dataset in datasets:
        for ranking_alg in ranking_algs:
            for group_definition in group_definitions:
                for labeled_pct in labeled_pcts:
                    for run in range(n_runs):

                        expt_config = {'dataset': dataset, 'learning_algs': ['ERM', 'IRM', 'GroupDRO', 'CORAL'],
                                    'group_definition': group_definition,
                                    'ranking_alg':  ranking_alg, 'labeled_pct': labeled_pct, 'run': run}
                                    
                        expt_configs.append(expt_config)

    # Note: This is only for ranking methods that operate on the predictions. Will have to add 
    # something for methods that  make use of the input features / embeddings, or outputted probabilities
    # Three types of inputs:
    # 1. accuracy matrix
    # 2. binary prediction matrix
    # 3. example features 
    # You can get 1. or 2. using get_model_values_df; 
    # we should be careful about making sure we don't mix up 1 and 2.
    # Another thing to be careful of: CivilComments does not have group attributes for unlabeled comments.

    # TODO: Record metrics for quality of estimated ranking; only doing kendalltau
    # TODO: Record group size
    # TODO: Something's wrong with CORAL, the sensitivities are really low; don't read into those results too much
    # TODO: Could move iteration over ranking algs to the inner loop, so that we're not regenerating the group vars each time
    # TODO: for certain ranking methods, we need not even load the unlabeled data (since this takes time)

    # Iterate over experiment configurations
    for config in tqdm(expt_configs):    
        np.random.seed(config['run'])

        # Load example binary predictions for all models on labeled + unlabeled split
        labeled_split_df = get_model_values_df(config, 'test', value='binary_prediction')
        # Set this to be test to speed up the runs but should load this conditionally based on the ranking method
        ## TODO: Change this to be unlabeled. 
        unlabeled_split_df = get_model_values_df(config, 'unlabeled', value='binary_prediction')

        # Add group information
        group_vars = config['group_definition'].split(',')  
        for group_var in group_vars:
            labeled_split_df[group_var] = labeled_split_df[group_var].swifter.progress_bar(False).apply(str)
            unlabeled_split_df[group_var] = unlabeled_split_df[group_var].swifter.progress_bar(False).apply(str)

        labeled_split_df['group'] = labeled_split_df[group_vars].swifter.progress_bar(False).apply(lambda x: ','.join(x), axis=1)
        unlabeled_split_df['group'] = ''
        # if config["dataset"] != 'civilcomments':
        #     unlabeled_split_df['group'] = unlabeled_split_df[group_vars].swifter.progress_bar(False).apply(lambda x: ','.join(x), axis=1)
        
        train_data_df = labeled_split_df.sample(frac = 0.5, random_state=config['run'])
        test_data_df = labeled_split_df.drop(train_data_df.index)

        # Subsample labeled data based on labeled_pct
        train_labeled_data_df = train_data_df.sample(frac = config['labeled_pct'], 
                                                    random_state=config['run'])

        # Rearrange to match function inputs
        algs = config['learning_algs']
        train_labeled_data = (train_labeled_data_df[algs].values, 
                            train_labeled_data_df['group'].values,
                            train_labeled_data_df['label'].values,)
        
        test_data = (test_data_df[algs].values, 
                            test_data_df['group'].values,
                            test_data_df['label'].values,)
        
        unlabeled_data = (unlabeled_split_df[algs].values,
                        unlabeled_split_df['group'].values)
        
        print("# Labeled training examples: ", train_labeled_data[0].shape[0])
        print("# Labeled test examples: ", test_data[0].shape[0])
        print("# Unlabeled examples: ", unlabeled_data[0].shape[0])

        # Estimate rankings per group
        if config['ranking_alg'] == 'random':
            groups, estimated_rankings, estimated_metrics = random_ranking(train_labeled_data, unlabeled_data)

        elif config['ranking_alg'] == 'labeled_global':
            groups, estimated_rankings, estimated_metrics = labeled_data_ranking(train_labeled_data, unlabeled_data, granularity="global")

        elif config['ranking_alg'] == 'labeled_group_specific':
            groups, estimated_rankings, estimated_metrics = labeled_data_ranking(train_labeled_data, unlabeled_data, granularity="group")

        elif  config['ranking_alg'] == 'oracle':
            groups, estimated_rankings, estimated_metrics = labeled_data_ranking(test_labeled_data, unlabeled_data)

        elif config['ranking_alg'] == 'dawid_skene':    
            # Not implemented yet
            groups, esimated_rankings, estimated_metrics = dawid_skene_ranking(train_labeled_data, unlabeled_data)
        
        assert len(groups) == len(estimated_rankings)
        _, true_rankings, true_metrics = labeled_data_ranking(test_data, unlabeled_data, granularity="group")
        # Create dataframe
        results = []


        for group, est_ranking, est_metrics, true_ranking, true_metric in zip(groups, estimated_rankings, estimated_metrics, 
                                                        true_rankings, true_metrics):
            result = {'group': group, 'ranking': est_ranking, 'metrics': est_metrics,
                    'true_ranking': true_ranking, 'true_metrics': true_metric
                    }
            result['kendalltau'] = kendalltau(true_metric, est_metrics)[0]
            result.update(config)
            results.append(result)
        
        # Save results
        results_df = pd.DataFrame(results)
        results_dir = create_results_path(config)
        os.makedirs(results_dir, exist_ok=True)
        results_df.to_csv(results_dir + '/' + get_config_hash(config) + '.csv')