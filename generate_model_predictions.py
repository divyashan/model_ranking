import os
import pickle
import numpy as np
import copy

import torch
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

from tqdm import tqdm
from scipy.special import softmax

from wilds import get_dataset
from wilds.common.data_loaders import get_eval_loader
from wilds.datasets.unlabeled.wilds_unlabeled_dataset import WILDSUnlabeledSubset

from wilds_local.examples.transforms import initialize_transform
from wilds_local.examples.models.initializer import initialize_model

from gpu_utils import restrict_GPU_pytorch
from paths import DATA_PATH, MODEL_PATH, create_predictions_path

restrict_GPU_pytorch('1')

algorithms = ['ERM', 'IRM', 'GroupDRO', 'CORAL']
algorithms = ['IRM']
datasets = ['civilcomments', 'camelyon17', 'ogbpcba']
splits = ["test", "unlabeled"]
splits = ["test"]

for dataset in datasets:
    print(dataset)
    for split in splits:
        
        # Load the config 
        wilds_config = pickle.load(open('../../' + dataset + '_config', 'rb'))

        transform = initialize_transform(wilds_config.transform, wilds_config, wilds_config.dataset, is_training=False)

        # TODO: Fix generation of predictions on unlabeled split
        # if 'unlabeled' in config['split'] :
        #     full_unlabeled_dataset = get_dataset(
        #             dataset=config['dataset'],
        #             root_dir=DATA_PATH,
        #             unlabeled=True
        #         )
            
        #     all_idxs = list(range(len(full_unlabeled_dataset)))
        #     test_data = WILDSUnlabeledSubset(full_unlabeled_dataset, all_idxs, transform=transform)

        full_dataset = get_dataset(dataset=dataset, root_dir=DATA_PATH)
        test_data = full_dataset.get_subset(split, transform=transform)
        test_loader = get_eval_loader("standard", test_data, batch_size=32, num_workers=2)

        ### Load model
        d_out = 2
        if dataset == "ogbpcba":
            d_out = 128
        model = initialize_model(wilds_config, d_out)

        for algorithm in algorithms:
            print(algorithm)

            model_path = MODEL_PATH + '/' + dataset + '/' + algorithm + '/best_model.pth'
            state_dict = torch.load(model_path)['algorithm']

            remove_prefix = "model."
            if algorithm == "CORAL":
                remove_prefix = "featurizer."
                state_dict = {k[len(remove_prefix):] if k.startswith(remove_prefix) else k: v for k, v in state_dict.items()}
                remove_prefix = "model.0."

            state_dict = {k[len(remove_prefix):] if k.startswith(remove_prefix) else k: v for k, v in state_dict.items()}
            model.load_state_dict(state_dict, strict=False)
            model = model.cuda()


            ### Generate predictions
            model.eval()
            all_y_true = []
            all_y_pred = []
            all_metadata = []
            i = 0
            for batch in tqdm(test_loader):
                # if "unlabeled" in "split":
                #     x, metadata = batch
                # else:
                x, y_true, metadata = batch
                    
                y_pred = model(x.cuda())
                all_y_pred.append(copy.deepcopy(y_pred.detach().cpu()))
                all_metadata.append(copy.deepcopy(metadata))

                if "unlabeled" not in split:
                    all_y_true.append(copy.deepcopy(y_true))

                i += 1

            all_y_pred = torch.concatenate(all_y_pred, axis=0)
            all_y_pred = softmax(all_y_pred.detach().cpu().numpy(), axis=1)
            all_metadata = torch.concatenate(all_metadata, axis=0)
            if "unlabeled" not in split:
                all_y_true = torch.concatenate(all_y_true, axis=0)

            ### Write out predictions
            config = {
                "dataset": wilds_config.dataset,
                "algorithm": algorithm,
                "split": split,
                "model": wilds_config.model,
            }
            predictions_path = create_predictions_path(config)
            os.makedirs(predictions_path, exist_ok=True)
            np.save(predictions_path + '/preds.npy', all_y_pred)
            np.save(predictions_path + '/metadata.npy', all_metadata)
            if "unlabeled" not in config["split"]:
                np.save(predictions_path + '/labels.npy', all_y_true)





algorithm = 'ERM'
split = "test"

# dataset = "camelyon17"
# model_name = 'densenet121'
dataset = "civilcomments"
model_name = "distilbert-base-uncased"

config = {
    "dataset": dataset,
    "algorithm": algorithm,
    "split": split,
    "model": model_name,
}

wilds_config = pickle.load(open('../../civilcomments_config', 'rb'))
transform = initialize_transform(wilds_config.transform, wilds_config, wilds_config.dataset, is_training=False)

if 'unlabeled' in config['split'] :
    full_unlabeled_dataset = get_dataset(
            dataset=config['dataset'],
            root_dir=DATA_PATH,
            unlabeled=True
        )
    
    all_idxs = list(range(len(full_unlabeled_dataset)))
    test_data = WILDSUnlabeledSubset(full_unlabeled_dataset, all_idxs, transform=transform)

else:
    full_dataset = get_dataset(dataset=dataset, root_dir=DATA_PATH)
    test_data = full_dataset.get_subset(split, transform=transform)

### Prepare the evaluation data loader
test_loader = get_eval_loader("standard", test_data, batch_size=32, num_workers=2)

### Load model
d_out = 2
if config["dataset"] == "ogbpcba":
    d_out = 128
model = initialize_model(wilds_config, d_out)

model_path = MODEL_PATH + '/' + config["dataset"] + '/' + config["algorithm"] + '/best_model.pth'
state_dict = torch.load(model_path)['algorithm']

remove_prefix = "model."
if config["algorithm"] == "CORAL":
    remove_prefix = "featurizer."
    state_dict = {k[len(remove_prefix):] if k.startswith(remove_prefix) else k: v for k, v in state_dict.items()}
    remove_prefix = "model.0."

state_dict = {k[len(remove_prefix):] if k.startswith(remove_prefix) else k: v for k, v in state_dict.items()}
model.load_state_dict(state_dict, strict=False)
model = model.cuda()

### Generate predictions
model.eval()
all_y_true = []
all_y_pred = []
all_metadata = []
i = 0
for batch in tqdm(test_loader):
    if "unlabeled" in config['split']:
        x, metadata = batch
    else:
        x, y_true, metadata = batch
        
    y_pred = model(x.cuda())
    all_y_pred.append(y_pred.detach().cpu())
    all_metadata.append(metadata)

    if "unlabeled" not in config['split']:
        all_y_true.append(y_true)

    i += 1

all_y_pred = torch.concatenate(all_y_pred, axis=0)
all_y_pred = softmax(all_y_pred.detach().cpu().numpy(), axis=1)
all_metadata = torch.concatenate(all_metadata, axis=0)
if "unlabeled" not in config["split"]:
    all_y_true = torch.concatenate(all_y_true, axis=0)

### Write out predictions
predictions_path = create_predictions_path(config)
os.makedirs(predictions_path, exist_ok=True)
np.save(predictions_path + '/preds.npy', all_y_pred)
np.save(predictions_path + '/metadata.npy', all_metadata)
if "unlabeled" not in config["split"]:
    np.save(predictions_path + '/labels.npy', all_y_true)
