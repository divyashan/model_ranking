import numpy as np
import random
from sklearn.metrics import confusion_matrix
import pdb
# Think about whether we should add the ensemble as one of the models we rank.

def compute_ranks(labeled_preds, labeled_targets, metric="sensitivity"):
    # Input: labeled_preds = N x M matrix 
    # Input: labeled_targets = N x 1 matrix 
    # Output: ranks = M x K matrix
    n_models = labeled_preds.shape[1]

    # This only works for binary predictions
    metrics = []
    

    for model_i in range(n_models):

        # Only one class observed for this group
        if len(set(labeled_targets)) == 1:
            if labeled_targets[0] == 1:
                sensitivity = len(labeled_preds[labeled_preds == 1]) / len(labeled_preds)
                specificity = 0
            else:
                specificity = len(labeled_preds[labeled_preds == 0]) / len(labeled_preds)
                sensitivity = 0

        # More than one class observed for this group
        else:
            tn, fp, fn, tp = confusion_matrix(labeled_targets, labeled_preds[:, model_i]).ravel()
            sensitivity = tp / (tp + fn)
            specificity = tn / (tn + fp)
            
        if metric == "sensitivity":
            metrics.append(sensitivity)
        elif metric == "specificity":
            metrics.append(specificity)

    ranks = np.argsort(metrics)
    
    return ranks

def random_ranking(labeled_data, unlabeled_data, rank_metric="sensitivity"):
    labeled_preds, labeled_groups, _ = labeled_data
    groups = sorted(list(set(labeled_groups)))
    n_models = labeled_preds.shape[1]

    model_choices = list(range(n_models))
    random_rankings = [tuple(random.sample(model_choices, len(model_choices))) for i in range(len(groups))]

    return groups, random_rankings

def labeled_data_ranking(labeled_data, unlabeled_data, rank_metric="sensitivity", granularity="group"):
    labeled_preds, labeled_groups, labeled_targets = labeled_data
    groups = sorted(list(set(labeled_groups)))
    
    group_specific_ranks = []
    if granularity == "group":
        for group in groups:
            group_labeled_preds, group_labeled_targets = labeled_preds[labeled_groups == group], labeled_targets[labeled_groups == group]
            model_ranks = compute_ranks(group_labeled_preds, group_labeled_targets, rank_metric)
            group_specific_ranks.append(tuple(model_ranks))
    elif granularity == "global":
        model_ranks = compute_ranks(labeled_preds, labeled_targets, rank_metric)
        group_specific_ranks = [tuple(model_ranks) for group in groups]

    return groups, group_specific_ranks

def dawid_skene_ranking(labeled_data, unlabeled_data, rank_metric="sensitivity"):
    labeled_preds,  labeled_groups, labeled_targets = labeled_data
    unlabeled_preds, unlabeled_groups = unlabeled_data
    
    if len(labeled_data) > 0:
        pass
        # Initialize priors
        
    # Produce N x M x K matrix 
    # Pass into dawid_skene 
    # Produce rankings
    
    
    # Subselect for the group? 
    
    pass

# This could be done on an example level or a group level
def nearest_neighbors_ranking(labeled_data, unlabeled_data, rank_metric="sensitivity"):
    labeled_preds,  labeled_groups, labeled_targets = labeled_data
    unlabeled_preds, unlabeled_groups = unlabeled_data
    # For each example in unlabeled_preds, find nearest neighbor in labeled_preds
    # Return model ranking for that example    

    pass

def walter_hui_ranking(labeled_data, unlabeled_data, rank_metric="sensitivity"):
    labeled_preds,  labeled_groups, labeled_targets = labeled_data
    unlabeled_preds, unlabeled_groups = unlabeled_data
    # Generate matrix of agreements and disagreements
    # Estimate alpha & beta for each model 
    
    pass