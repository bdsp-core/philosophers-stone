import sys
import os
import math
import torch
import random
import numpy as np
from pathlib import Path
from pytorch_lightning import LightningModule, Trainer, seed_everything, Callback
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor, RichProgressBar
from pytorch_lightning.callbacks.progress.rich_progress import RichProgressBarTheme # from lightning.pytorch
from sklearn.model_selection import train_test_split
import shutil 
import pandas as pd
from torchsummary import summary
import time
import itertools
import gc
from oracle_model import *
from maxxvit_oracle_v2_3 import *

import warnings
warnings.filterwarnings("ignore", message="Checkpoint directory .* exists and is not empty")

import logging

logger = logging.getLogger('pytorch_lightning.utilities.rank_zero')
class IgnorePLFilter(logging.Filter):
    def filter(self, record):
        return 'available:' not in record.getMessage()
logger.addFilter(IgnorePLFilter())
class SuppressCheckpointWarningFilter(logging.Filter):
    def filter(self, record):
        return "Checkpoint directory" not in record.getMessage()
logger.addFilter(SuppressCheckpointWarningFilter())

from model_evaluation import *
from dataset import *
import argparse
import re
import copy

PATH_DATASETS = os.environ.get("PATH_DATASETS", ".")
torch.cuda.empty_cache()
gc.collect();

SCRIPT_DIR = Path(__file__).resolve().parent
TRAINING_OUTPUT_DIR = SCRIPT_DIR / "training_output"
TRAINING_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def evaluate_model_checkpoint(path_ckpt, title, savedir, model_name, 
                                n_covariates, 
                                train_ds, validation_ds, test_ds, 
                                regression_targets, classification_targets, 
                                table, 
                                device,
                                indices=None,
                                save_model_predictions_flag=False, BATCH_SIZE=1,
                                ):

        ckpt_name = os.path.basename(path_ckpt)
        print(f"\033[94mEvaluate model checkpoint: {ckpt_name}\033[0m")
        model = init_model(model_name, n_covariates=n_covariates) # , lr=lr)
        assert os.path.exists(path_ckpt), f"Checkpoint file does not exist: {path_ckpt}"
        model = model.load_from_checkpoint(path_ckpt)
        # model.eval() # activaing this changes/reduces performance?

        # Evaluation for training, validation, and test sets
        y_regression_train, yp_regression_train, y_classification_train, yp_classification_train, y_annotations_train, yp_annotations_train, features_lhl_train = \
            predict_routine(model, train_ds, BATCH_SIZE, dset_name='train', device=device)
        y_regression_validation, yp_regression_validation, y_classification_validation, yp_classification_validation, y_annotations_validation, yp_annotations_validation, features_lhl_validation = \
            predict_routine(model, validation_ds, BATCH_SIZE, dset_name='valid', device=device)
        y_regression_test, yp_regression_test, y_classification_test, yp_classification_test, y_annotations_test, yp_annotations_test, features_lhl_test = \
            predict_routine(model, test_ds, BATCH_SIZE, dset_name='test ', device=device)

        # Consolidate results
        results_regression = consolidate_results(y_regression_train, yp_regression_train, y_regression_validation, yp_regression_validation, 
                                                 y_regression_test, yp_regression_test, string="regression")
        results_classification = consolidate_results(y_classification_train, yp_classification_train, y_classification_validation, yp_classification_validation, 
                                                     y_classification_test, yp_classification_test, string="classification")

        # LHL dictionary
        results_features_lhl = {
            "features_lhl_train": features_lhl_train,
            "features_lhl_validation": features_lhl_validation,
            "features_lhl_test": features_lhl_test,
        }

        # Combine regression and classification results for analysis
        for key in results_regression.keys():
            key_classification = key.replace('regression', 'classification')
            results_regression[key] = np.concatenate([results_regression[key], results_classification[key_classification]], axis=1)

        results_annotations = consolidate_annotations(y_annotations_train, yp_annotations_train, y_annotations_validation, yp_annotations_validation, y_annotations_test, yp_annotations_test)

        if save_model_predictions_flag:
            file_ids = table.fileid.values
            
            save_model_predictions(regression_targets, classification_targets, results_regression, results_features_lhl, file_ids, savedir, title, indices)

            if 0:
                results = {**results_regression, **results_annotations}
                pickle.dump(results, open(os.path.join(savedir, f"results_predictions_{ckpt_name.replace('.ckpt', '')}.p"), 'wb'))

        return results_regression, results_classification, results_annotations


def consolidate_results(y_train, yp_train, y_val, yp_val, y_test, yp_test, string=""):
    return {
        f"y_{string}_train": y_train,
        f"yp_{string}_train": yp_train,
        f"y_{string}_validation": y_val,
        f"yp_{string}_validation": yp_val,
        f"y_{string}_test": y_test,
        f"yp_{string}_test": yp_test,
    }


def consolidate_annotations(y_annotations_train, yp_annotations_train, y_annotations_validation, yp_annotations_validation, y_annotations_test, yp_annotations_test):
    return {
        "y_annotations_train": y_annotations_train,
        "yp_annotations_train": yp_annotations_train,
        "y_annotations_validation": y_annotations_validation,
        "yp_annotations_validation": yp_annotations_validation,
        "y_annotations_test": y_annotations_test,
        "yp_annotations_test": yp_annotations_test,
    }


def save_model_predictions(regression_targets, classification_targets, results_regression, results_features_lhl, file_ids, savedir, title, indices=None):

    title = 'last' if 'last' in title else title.split('sleep_oracle-epoch=')[-1] + 'epoch'

    if indices is None:
        indices = {'all': np.arange(len(file_ids))}

    for set_name, idx_sel in indices.items():
        results_cols = [f"{target}_true" for target in regression_targets + classification_targets] + \
                       [f"{target}_pred" for target in regression_targets + classification_targets]
        
        lhl_cols = [f"lhl_{i}" for i in range(results_features_lhl[f'features_lhl_{set_name}'].shape[1])]

        df_results = pd.DataFrame(columns=results_cols + lhl_cols)

        for i, target in enumerate(regression_targets + classification_targets):
            df_results[target + '_true'] = results_regression[f'y_regression_{set_name}'][:, i]
            df_results[target + "_pred"] = results_regression[f'yp_regression_{set_name}'][:, i]
        
        for i in range(results_features_lhl[f'features_lhl_{set_name}'].shape[1]):
            df_results[lhl_cols[i]] = results_features_lhl[f'features_lhl_{set_name}'][:, i]
            
        df_results.index = file_ids[idx_sel]
        df_results.to_csv(os.path.join(savedir, f"df_results_{set_name}_{title}.csv"))


def get_min_loss_entry(list_str):
    """
    Find checkpoint file with minimum loss
    """
    # Regex pattern to find the loss value
    pattern = re.compile(r"val_loss_lhl=([0-9.]+)")
    
    min_loss = float('inf')  # initialize minimum loss as infinity
    min_loss_entry = ""  # initialize entry with minimum loss as empty string
    
    for entry in list_str:
        match = pattern.search(entry)
        if match:
            loss_str = match.group(1)
            # Remove trailing decimal point if present
            if loss_str[-1] == '.':
                loss_str = loss_str[:-1]
            loss = float(loss_str)
        
            if loss < min_loss:
                min_loss = loss
                min_loss_entry = entry
    
    return min_loss_entry


class CustomRichProgressBar(RichProgressBar):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def init_layout(self):
        """Override this method to customize the progress bar layout."""
        self.progress = self.main_panel.add_widget(self._get_progress_bar())
        self.metrics = self.main_panel.add_widget(self._get_metrics_table())

    def get_metrics(self, trainer, model):
        # don't show the version number
        items = super().get_metrics(trainer, model)
        items.pop("v_num", None)
        remove_items = []
        for item in items:
            if 'step' in item:
                remove_items.append(item)
        for item in remove_items:
            items.pop(item, None)
        return items
    
    

progress_bar = CustomRichProgressBar(
    leave=True,
    theme=RichProgressBarTheme(
        description="green_yellow",
        progress_bar="green1",
        progress_bar_finished="green1",
        progress_bar_pulse="#6206E0",
        batch_progress="green_yellow",
        time="grey82",
        processing_speed="grey82",
        metrics="gold1",
    )
)


def xavier_init(model):
    for name, param in model.named_parameters():
        if not name.startswith("stem.conv") or not name.startswith("head"):
            continue
        if name.endswith(".bias"):
            param.data.fill_(0)
        else:
            bound = math.sqrt(6) / math.sqrt(param.shape[0] + param.shape[1])
            param.data.uniform_(-bound, bound)
            
def init_model(model_name:str, n_covariates=None, lr=2e-2, cost_function='mae', n_targets_regression=1, weight_regression_targets=None, n_targets_classification=0,
               weight_classification=None, target_weights_classification=None, weight_stage=None, n_train=1, lr_gamma=0.999, regression_pre_final_dim=1024,
               fs_time=1, target_names:list=[], target_output_dims:list=[], task_types:list=[]):
    """Initialize model based on model_name"""

    if target_weights_classification is None:
        # default: equal weights for all classification targets. Especially for TL, this can be changed.
        target_weights_classification = torch.tensor([1.0] * n_targets_classification, dtype=torch.float32)

    if model_name == 'MaxxVit_oracle_v2_3':

        model = MaxxVit_oracle_v2_3(lr, cost_function, n_targets_regression, weight_regression_targets, 
                                    n_targets_classification, weight_classification, target_weights_classification,
                                    weight_stage, n_train, lr_gamma, n_covariates,
                                    target_names, target_output_dims, task_types,
                                    dim_final_latent_space=regression_pre_final_dim,
                                    fs_time=fs_time)

    else:
        raise ValueError(f"model_name {model_name} not implemented")

    xavier_init(model)

    return model

def save_fct_params(vars_local):
    dict_params = {}
    for key, value in vars_local.items():
        if not key.startswith("__") and not callable(value):
            dict_params[key] = value
    dict_params = copy.deepcopy(dict_params)
    dict_params['path_mastertable_full'] = dict_params['path_mastertable']
    dict_params['path_mastertable'] = dict_params['path_mastertable'].split('_table')[0]
    
    return dict_params


def process_targets(table, regression_targets, weight_regression_targets, classification_targets):
    
    # if cogd (cognitive domains) is in regression targets (which is a simplification for calling this functionatlity), then add the domain targets here actually. also the weight needs to be modified.
    # if any([x.startswith('cog') for x in regression_targets]):
    if ('cog-all' in regression_targets) or ('cog-main' in regression_targets):

        if 'cog-all' in regression_targets:
            cog_target = 'cog-all'
            df_cog_vars = pd.read_csv('cognitive_vars_all.csv')
        elif 'cog-main' in regression_targets:
            cog_target = 'cog-main'
            df_cog_vars = pd.read_csv('cognitive_vars_main.csv')
        else:
            raise ValueError(f'Unknown cognition variable in regression targets {regression_targets}')

        n_cohorts_cognition = df_cog_vars.shape[1]
        vars_cognition = df_cog_vars.values.flatten()
        vars_cognition = [x for x in vars_cognition if pd.notna(x)]
        mean_n_vars_per_cohort = np.round(len(vars_cognition) / n_cohorts_cognition, 2)

        idx_cog_target = regression_targets.index(cog_target)
        regression_targets.remove(cog_target)
        regression_targets = regression_targets + vars_cognition

        weight_cognition = weight_regression_targets[idx_cog_target]
        weight_regression_targets.pop(idx_cog_target)
        weight_cognition = weight_cognition / mean_n_vars_per_cohort
        weight_regression_targets = weight_regression_targets + [weight_cognition] * len(vars_cognition)

    if 'dx' in classification_targets:
        # disease cols dx-tm (disease target-matched) have 1 for disease, 0 for negative matched control and NaN otherwise).
        disease_cols = [x for x in table.columns if x.startswith('dx-tm')]
        classification_targets.remove('dx')
        classification_targets = disease_cols
        n_targets_classification = len(classification_targets)
    elif any(['dx' in x for x in classification_targets]):
        n_targets_classification = len(classification_targets)
        # assert the classification_targets are in the table columns:
        assert all([x in table.columns for x in classification_targets]), "Classification targets not in table columns"
    else:
        classification_targets = None
        n_targets_classification = 0

    if 'self' in regression_targets:
        selfsupervised_cols = [x for x in table.columns if x.startswith('f_')]
        # analog logic as for cog_domains above:
        idx_selfsupervised = regression_targets.index('self')
        regression_targets.remove('self')
        regression_targets = regression_targets + selfsupervised_cols

        weight_selfsupervised = weight_regression_targets[idx_selfsupervised]
        weight_regression_targets.pop(idx_selfsupervised)
        weight_selfsupervised = weight_selfsupervised / len(selfsupervised_cols)
        weight_regression_targets = weight_regression_targets + [weight_selfsupervised] * len(selfsupervised_cols)

    return regression_targets, weight_regression_targets, classification_targets, n_targets_classification



def main(gpu_id, path_mastertable, model_name, regression_targets, weight_regression_targets, classification_targets,  weight_classification,
         covariates, fold_id, cost_function_regression, lr,  weight_stage, min_epochs, max_epochs, n_training_samples, lr_gamma, regression_pre_final_dim,
         fs_time, transfer_learning, target_tl, pre_trained_ckpt_path, pre_trained_ckpt_type, resume_training,
         evaluation_only, eval_ckpt_dir, print_main_info):

    if gpu_id == 'cpu':
        accelerator = 'cpu'
        devices = 1
        device = torch.device('cpu')
        print("Using CPU for training.")
        BATCH_SIZE = 1
        precision ="32-true"
    
    else:
        AVAIL_GPUS = [int(gpu_id)] 
        BATCH_SIZE = 1 * len(AVAIL_GPUS)
        print(f"GPUs: {AVAIL_GPUS}")
        
        torch.cuda.empty_cache()
        gc.collect();
        device = torch.device(f'cuda:{AVAIL_GPUS[0]}')
        accelerator = 'gpu'
        devices = AVAIL_GPUS
        precision = "32-true" # "32-true" # or "16-mixed"
        
        torch.manual_seed(42 + AVAIL_GPUS[0])
        random.seed(42 + AVAIL_GPUS[0])
        np.random.seed(42 + AVAIL_GPUS[0])
        seed_everything(42 + AVAIL_GPUS[0], workers=True); 

    log_root_dir = TRAINING_OUTPUT_DIR
    dict_params = save_fct_params(locals())
    
    table = load_master_table(path_mastertable)
    
    if target_tl is not None:
        assert target_tl in table.columns, f"Target transfer learning {target_tl} not in table columns"
        # select only the rows where the target_tl is not NaN:
        table = table[table[target_tl].notna()]
        table = table.reset_index(drop=True)
        if print_main_info: print(f"TL. Table shape after removing NaNs for target {target_tl}: {table.shape}")

    if 1: # use fold_id from table to define train+validation and test set, use stratified fold to divide training into training and validation sets:
        idx_train = table[table['fold_id'] != fold_id].index.values
        idx_test = table[table['fold_id'] == fold_id].index.values
        # split of training set into training and validation set, stratified by stratify_quintile:
        idx_train, idx_validation = train_test_split(idx_train, test_size=0.2, random_state=42, stratify=table.loc[idx_train, 'stratify_quintile'])
        # always shuffle the idx_train and idx_validation, to speed up training when parallelizing:
        idx_train = np.random.permutation(idx_train)
        idx_validation = np.random.permutation(idx_validation)

    
    print("Len table:", len(table))
    print(list(table['fileid'].values[:3]))
    
    n_train = len(idx_train)
    table_id = path_mastertable.split('_table')[0]
    table_id = table_id.replace('.csv', '')
    
    assert len(table_id) > 0, "path_mastertable is supposed to be named like 'master_table-[...]' or 'pretrain_table-[...]'"
    
    if transfer_learning:
        target_string = f"TL-{target_tl}"
    else:
        target_string = regression_targets
    
    run_id = f"{model_name.replace('MaxxVit_oracle_', '').replace('Oracle_', '')}_{table_id}_{target_string}_{covariates}_{cost_function_regression}_" + \
                f"wSS{weight_stage}_wRe{str(weight_regression_targets)}_wCl{weight_classification}" + \
                f"_ntr{n_train}_lr{lr}-{lr_gamma}_d{regression_pre_final_dim}_ep{max_epochs}_f{fold_id}"
    run_id = run_id.replace(', ', '_')
    
    accumulate_grad_batches = 1
    if accumulate_grad_batches > 1:
        run_id += f"_ab{accumulate_grad_batches}"
        lr = lr * accumulate_grad_batches
        
    dict_params['run_id'] = run_id
    
    log_dir_lightning = os.path.join(log_root_dir, f"{run_id}")
    skip_min_val_determination = False
    # now print the run id once more, if it is shell then it shall be printed in blue:
    print(f"\n\033[31mrun_id={run_id}\033[0m\n")
    
    if not os.path.exists(log_dir_lightning):
        os.makedirs(log_dir_lightning)

    if print_main_info: print('log_root_dir', log_root_dir)
    if print_main_info: print('log_dir_lightning', log_dir_lightning)
    
    # Save the parameters to a file
    with open(os.path.join(log_dir_lightning, "params.p"), "wb") as f:
        pickle.dump(dict_params, f)
        
    gc.collect();
    
    regression_targets, weight_regression_targets, classification_targets, n_targets_classification = process_targets(
        table, regression_targets, weight_regression_targets, classification_targets
    )

    # save the regression targets and classification targets, and pre_trained_ckt_path to a file:
    pd.DataFrame(regression_targets, columns=['regression_targets']).to_csv(os.path.join(log_dir_lightning, 'regression_targets.csv'), index=False)
    pd.DataFrame(classification_targets, columns=['classification_targets']).to_csv(os.path.join(log_dir_lightning, 'classification_targets.csv'), index=False)
    if transfer_learning:
        
        with open(os.path.join(log_dir_lightning, 'pre_trained_ckpt_path.txt'), 'w') as f:
            f.write(pre_trained_ckpt_path)
            
    weight_regression_targets = torch.tensor(weight_regression_targets, dtype=torch.float32).to(device)

    n_targets_regression = len(regression_targets)

    if print_main_info: 
        print('N targets regression = ', n_targets_regression)
        print('N targets classification = ', n_targets_classification)
        print('classification targets = ', classification_targets)
        # print dx-tm-dementia value_counts of training, va and test set:
        if target_tl is not None:
            print('TL target:', target_tl)
        for dx_tmp in ['dx-tm-dementia', 'dx-tm-depression']:
            print(f'{dx_tmp} value_counts:')
            print('train', table.loc[idx_train, dx_tmp].value_counts())
            print('validation', table.loc[idx_validation, dx_tmp].value_counts())
            print('test', table.loc[idx_test, dx_tmp].value_counts())
    
    target_weights_classification = torch.tensor([1.0] * n_targets_classification, dtype=torch.float32)
    
    if transfer_learning:
        
        # double checks. If 'dx' is in target_tl, then the classification weight should be non-zero:
        if 'dx' in target_tl:
            assert target_weights_classification.sum() > 0, "Classification weights are zero, but 'dx' is in target_tl"
        # if 'cog' or 'np' is in target_tl, then the regression weight should be non-zero:
        if (target_tl.startswith('cog')) or (target_tl.startswith('np')):
            assert weight_regression_targets.sum() > 0, "Regression weights are zero, but 'cog' or 'np' is in target_tl"
            
        assert pre_trained_ckpt_path is not None, "Pre-trained checkpoint path is required for transfer learning"
        if not os.path.exists(pre_trained_ckpt_path): # typically run_id is passed, not full root dir.
            pre_trained_ckpt_path = os.path.join(log_root_dir, pre_trained_ckpt_path)
        # replace '1e-04' with '0.0001' in pre_trained_ckpt_path:
        pre_trained_ckpt_path = pre_trained_ckpt_path.replace('1e-04', '0.0001')

        assert os.path.isdir(pre_trained_ckpt_path), f"Pre-trained checkpoint directory does not exist: {pre_trained_ckpt_path}"
        # load regression and classification targets table from base model training:
        regression_targets_base_model = list(pd.read_csv(os.path.join(pre_trained_ckpt_path, 'regression_targets.csv')).regression_targets.values)
        classification_targets_base_model = list(pd.read_csv(os.path.join(pre_trained_ckpt_path, 'classification_targets.csv')).classification_targets.values)
        
    dset = Spectrogram_1_2_Dataset(path_mastertable=path_mastertable, 
                                    regression_targets=regression_targets,
                                    classification_targets=classification_targets,
                                    covariates=covariates,
                                    target_tl=target_tl)
    
    if print_main_info: 
        print(f"len(dataset) = {len(dset)}")
        print(dset[0][0].shape)
    

    assert len(dset) == len(table), f"Mismatch not expected, needs fix, len(dset)={len(dset)}, len(table)={len(table)}"
    if print_main_info: print(f"N training samples = {n_train}")

    train_ds = Subset(dset, idx_train)
    validation_ds = Subset(dset, idx_validation)
    test_ds = Subset(dset, idx_test)
    if print_main_info: print(f"Samples TR/VA/TE: {len(train_ds)}/{len(validation_ds)}/{len(test_ds)}")

    target_names = regression_targets + classification_targets
    target_output_dims = [1] * len(target_names)
    task_types = ['regression'] * len(regression_targets) + ['classification'] * len(classification_targets)
        
    # MODELLING

    n_covariates = len(covariates)
    if 'cohort' in covariates: n_covariates += 6  # one-hot encoding of cohort
    if print_main_info: print('model init, weight_regression_targets', len(weight_regression_targets))
    if 0: print(target_names)
    model = init_model(model_name, n_covariates=n_covariates, lr=lr, cost_function=cost_function_regression, 
                        n_targets_regression=n_targets_regression, weight_regression_targets=weight_regression_targets,
                        n_targets_classification=n_targets_classification, weight_classification=weight_classification,
                        target_weights_classification=target_weights_classification,
                        weight_stage=weight_stage, n_train=n_train, lr_gamma=lr_gamma, regression_pre_final_dim=regression_pre_final_dim,
                        fs_time=fs_time, target_names=target_names, target_output_dims=target_output_dims, task_types=task_types)
    
    if print_main_info: 
        print(summary(model, [train_ds[0][0].shape, train_ds[0][4].shape], device=device))  # model.cuda()

    pdf_filepaths = {
            'logger': os.path.join(log_root_dir, f'logging_viz/pdf/{run_id}.pdf'),
        }

    train = True
    if evaluation_only:
        train = False
        
    if train is False:
        if print_main_info: 
                print("\n\nTraining is disabled. Only evaluation of model checkpoints is performed.")

    if train:

        # Init DataLoader
        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, num_workers=3,
                                    persistent_workers=True)
        validation_loader = DataLoader(validation_ds, batch_size=BATCH_SIZE, num_workers=3,
                                        persistent_workers=True)
        


        save_top_k = 11 # 20 # 5 # save_top_k: paramter for ModelCheckpoint, saves the best k models according to the metric monitored.
        every_n_epochs = 1 # 5 # every_n_epochs: parameter for ModelCheckpoint, saves a checkpoint every n epochs.
            

        checkpoint_val_loss = ModelCheckpoint(
            monitor="val_loss_lhl_epoch", # lhl: combination of regression+classification loss.
            dirpath=log_dir_lightning,
            filename="sleep_oracle-{epoch:02d}-{val_loss_lhl:.4f}",
            save_top_k=save_top_k,
            save_last=True,
            mode="min",
            )
                
        checkpoint_epoch = ModelCheckpoint(
            every_n_epochs=every_n_epochs,
            monitor="epoch",
            dirpath=log_dir_lightning,
            filename="sleep_oracle-{epoch:02d}-{val_loss_lhl:.4f}",
            save_top_k=1000,
            mode='max',
        )

        callbacks = [checkpoint_val_loss, checkpoint_epoch]

        # early stopping based on validation loss
        early_stopping_activated = False
        if early_stopping_activated:
            early_stopping = EarlyStopping("val_loss_lhl_epoch",
                                            patience=20,
                                            )
            callbacks.append(early_stopping)

        lr_monitor = LearningRateMonitor(logging_interval="step")
        callbacks.append(lr_monitor)
        callbacks.append(progress_bar)

        if print_main_info: print(f"min_epochs={min_epochs}\nmax_epochs={max_epochs}")
        if transfer_learning:
            if not os.path.exists(os.path.join(log_dir_lightning, 'lightning_logs')):
                os.makedirs(os.path.join(log_dir_lightning, 'lightning_logs'))

        check_val_every_n_epoch = 50
        trainer = Trainer(
            accelerator = accelerator,
            devices=devices,
            min_epochs=min_epochs,
            max_epochs=max_epochs,
            log_every_n_steps=1,
            deterministic=False,  # reproducibility, somehow throws error sometimes
            callbacks=callbacks,
            default_root_dir=log_dir_lightning,
            fast_dev_run=False,
            gradient_clip_val=0.5,
            enable_model_summary=not transfer_learning,
            accumulate_grad_batches=accumulate_grad_batches,
            check_val_every_n_epoch=check_val_every_n_epoch,
            precision=precision,
        )

        # TRAIN oracle_model ⚡

        ckpt_path = None
        trainer.fit(model, train_loader, validation_loader, ckpt_path=ckpt_path)


        # Visualize training progress / logger info (tensorboard) (imported from model_evaluation.py):
        log_dir = trainer.logger.log_dir
        if train == False:
            log_dir = log_dir.replace('version_1', 'version_0')
        log_files = os.listdir(log_dir)
        events_file = [file for file in log_files if 'events.out.tfevents' in file]
        if len(events_file) == 0:
            print("No events file found in log_dir")
        elif len(events_file) > 0:
            if len(events_file) > 1:
                print("More than one events file found in log_dir")
            path_eventfile = os.path.join(log_dir, events_file[0])
        plot_train_val_loss_lr(path_eventfile, filename=run_id)
        
        
    # TEST (performance evaluation & plots):

    ckpt_dir = log_dir_lightning
    ckpts = [x for x in os.listdir(ckpt_dir) if ".ckpt" in x]
            
    if evaluation_only:
        if print_main_info: print("Evaluation only")
        if eval_ckpt_dir is not None:
            ckpt_dir = eval_ckpt_dir
            ckpts = [x for x in os.listdir(ckpt_dir) if ".ckpt" in x]


    ckpt_min_validation_loss = get_min_loss_entry(ckpts)
    regression_targets_to_evaluate = regression_targets
    classification_targets_to_evaluate = classification_targets

    ckpts = sorted(ckpts)
    
    for ckpt in ckpts:
        title = f"{run_id}_{ckpt.replace('.ckpt', '')}"
        if ckpt == ckpt_min_validation_loss:
            title += "_minval"
            save_model_predictions_flag = True
        else:
            save_model_predictions_flag = False
        if 'master_mp' in path_mastertable:
            save_model_predictions_flag = True

        path_ckpt = os.path.join(ckpt_dir, ckpt)
        if print_main_info: print('path_ckpt', path_ckpt)
        
        indices = {'train': idx_train, 'validation': idx_validation, 'test': idx_test}
        # Call the evaluation function
        results_regression, results_classification, results_annotations = \
            evaluate_model_checkpoint(
                path_ckpt, title, log_dir_lightning, model_name, n_covariates, train_ds, validation_ds, test_ds,
                regression_targets_to_evaluate, classification_targets_to_evaluate, table, device, indices=indices, 
                save_model_predictions_flag=save_model_predictions_flag,
            )

        verbose = True
        
        results_correlation = plot_regression_results(results_regression.copy(), regression_targets_to_evaluate + classification_targets_to_evaluate, title=title, 
                                savedir=log_dir_lightning, cost_function_regression=cost_function_regression, 
                                verbose=verbose)
        
        self_supervised_contained = any([x.startswith('f_') for x in regression_targets_to_evaluate])
        if self_supervised_contained:
            results_correlation = plot_regression_results(results_regression.copy(), regression_targets_to_evaluate + classification_targets_to_evaluate, title=title, 
                            savedir=None, cost_function_regression=cost_function_regression, 
                            plot_selfsupervised=True, verbose=verbose)
        
        pdf_filepaths['regression_scatter'] = os.path.join(log_root_dir, 'results_regression/pdf/scatter_' + title + '.pdf')

        plot_regression_heatmap_r(results_correlation.copy(), title, regression_targets_to_evaluate + classification_targets_to_evaluate)

        pdf_filepaths['regression_heatmap_r'] = os.path.join(log_root_dir, 'results_regression/pdf/regression_heatmap_r_' + title + '.pdf')
        
        results_annotations = evaluate_sleep_stage_performance_cm_and_kappa(results_annotations)
        plot_confusion_matrices(results_annotations, title=title, savedir=log_dir_lightning)
        pdf_filepaths['ss_confusion'] =  os.path.join(log_root_dir, f'results_ss_cm/pdf/cm_3x2_{title}.pdf')

        save_results_csv([results_correlation.copy(),
                            results_annotations['cohen_kappa_train'],
                            results_annotations['cohen_kappa_validation'],
                            results_annotations['cohen_kappa_test']],
                            title,
                            regression_targets_to_evaluate + classification_targets_to_evaluate,
                            dict_params=dict_params)


if __name__ == "__main__":

    # Parse arguments:
    parser = argparse.ArgumentParser(description='Train sleep oracle model')
    parser.add_argument('gpu_id', help='GPU ID (e.g. 0)')
    parser.add_argument('path_mastertable', help='Path to master table (e.g. "master_table_2021-07-01.csv")')
    parser.add_argument('model_name', help='Model name (e.g. "SimpleUNet", "SimpleCNN", "MaxxVit_oracle_v2", "MaxxVit_oracle_v2_1")')
    parser.add_argument('--fold_id', type=int, help='Fold ID (e.g. 0)')
    parser.add_argument('--regression_targets', nargs='+', type=str, help="Regression targets to predict, as list (e.g. 'age_z', 'cog_fluid')")
    parser.add_argument('--weight_regression_targets', nargs='+', type=float, help='Weight of regression tasks (target argument). Has to have same number of elements as argument target and sum up to 1')
    parser.add_argument('--classification_targets', nargs='+', type=str, help="Classification targets to predict, as list (e.g. 'dx'")
    parser.add_argument('--weight_classification', type=float, default=0.3, help="Cost function weight for classification (diseases) task")
    parser.add_argument('--covariates', nargs='+', type=str, default='', help='Covariates to include for final classification head (e.g. "age, sex, cohort, apoe)')
    parser.add_argument('--cost_function_regression', default="rmse", help='Cost function for regression(e.g. "rmse", "mae", "rmse_pearson", "mae_pearson")')
    parser.add_argument('--learning_rate', type=float, default=5e-4, help="Learning rate")
    parser.add_argument('--weight_stage', type=float, default=0.3, help="Cost function weight for sleep staging task")
    parser.add_argument('--min_epochs', type=int, default=5, help="Minimum number of epochs")
    parser.add_argument('--max_epochs', type=int, default=200, help="Maximum number of epochs")
    parser.add_argument('--n_training_samples', type=int, default=0, help="Number of training samples. Default 0: use all as specified in table and folds.")
    parser.add_argument('--lr_gamma', type=float, default=0.999, help="Gamma parameter for cyclic LR. Smaller value faster decrease.")
    parser.add_argument('--regression_pre_final_dim', type=int, default=1024, help="Dimension of pre-final layer (last hidden layer) for regression tasks.")
    parser.add_argument('--fs_time', type=int, default=1, help="Spectrogram time resolution")
    parser.add_argument('--transfer_learning', type=lambda x: (str(x).lower() == 'true'), default=False, help='Enable or disable transfer learning (default: %(default)s)')
    parser.add_argument('--target_tl', type=str, default=None, help='Target for transfer learning. Currently expected to be one variable')
    parser.add_argument('--pre_trained_ckpt', type=str, default=None, help='Path to the pre-trained model checkpoint for transfer learning')
    parser.add_argument('--pre_trained_ckpt_type', type=str, default='minval', help='If "pre_trained_ckpt" is a directory (and not a .ckpt), specify the type of checkpoint to use (e.g. "minval", "last")')
    parser.add_argument('--resume_training', type=lambda x: (str(x).lower() == 'true'), default=False, help='Enable or disable resume training (default: %(default)s)')
    parser.add_argument('--evaluation_only', type=lambda x: (str(x).lower() == 'true'), default=False, help='Enable or disable training (default: %(default)s)')
    parser.add_argument('--eval_ckpt_dir', type=str, default=None, help='Path to the directory containing the model checkpoints to evaluate')
    parser.add_argument('--print_main_info', type=lambda x: (str(x).lower() == 'true'), default=False, help='Print main info or not (default: %(default)s)')

    args = parser.parse_args()
    assert sum(args.weight_regression_targets) == 1, "Convention: weight_regression_targets must sum to 1. Otherwise balance with SS task weight and LR is not correct/stable."
    # assert args.model_name in ["SimpleUNet", "SimpleCNN", "Sleep_Oracle_MLT_4_v2", "MaxxVit_oracle_v1", "MaxxVit_oracle_v2", "MaxxVit_oracle_v2_1", "MaxxVit_oracle_v2_1_fs2", "MaxxVit_oracle_v2_2", "MaxxVit_oracle_v2_3", "MaxxVit_oracle_v2_4", "MaxxVit_oracle_v2_5", "MaxxVit_oracle_v2_6"], "model_name not in list"
    assert len(args.regression_targets) == len(args.weight_regression_targets), f"Mismatch #targets and #target_weights ({args.regression_targets} vs {args.weight_regression_targets})"
    assert os.path.exists(args.path_mastertable), f"Path to master table does not exist: {args.path_mastertable}"

    main(args.gpu_id, 
         args.path_mastertable,
         args.model_name, 
         args.regression_targets, 
         args.weight_regression_targets,
         args.classification_targets,
         args.weight_classification,
         args.covariates, 
         args.fold_id,
         args.cost_function_regression, 
         args.learning_rate, 
         args.weight_stage,
         args.min_epochs,
         args.max_epochs,
         args.n_training_samples,
         args.lr_gamma,
         args.regression_pre_final_dim,
         args.fs_time,
        args.transfer_learning,
        args.target_tl,
        args.pre_trained_ckpt,      # required both for transfer learning, resume-training and evaluation.
        args.pre_trained_ckpt_type,
        args.resume_training,    # if True, then resume training from the latest checkpoint in the log_dir_lightning directory or a specific checkpoint. The difference to TL is that in "TL", we assume the model is finetuned to a specific new task/target. Here, the same task is continued (although potentially with new data/table).
        args.evaluation_only,
        args.eval_ckpt_dir,
        args.print_main_info,
        )
# E.g. python step5_train_model.py 0 MaxxVit_oracle_v2_1  --fold_id 0 --regression_targets 'age_z' 'cog_fluid' --weight_regression_targets 0.5 0.5 --cost_function_regression 'rmse' --learning_rate 5e-4 --weight_stage 0.5 --min_epochs 5 --max_epochs 5 --n_training_samples 0
