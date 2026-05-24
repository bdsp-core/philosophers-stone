import os
import torch
torch.manual_seed(42)
import random
random.seed(69)
import numpy as np
np.random.seed(42)
from tqdm import tqdm
import re
import seaborn as sns
from scipy.stats import pearsonr, linregress

from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, Subset
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from torchmetrics import Accuracy
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from itertools import product
from filelock import FileLock


import pickle
from pycm import ConfusionMatrix
import PyPDF2

import warnings

warnings.filterwarnings("ignore", message="invalid value encountered in sqrt")
warnings.filterwarnings("ignore", message="invalid value encountered in double_scalars")
warnings.filterwarnings("ignore", message="Mean of empty slice")
warnings.filterwarnings("ignore", message="invalid value encountered in true_divide")
warnings.filterwarnings("ignore", category=RuntimeWarning, message="An input array is nearly constant")


PATH_DATASETS = os.environ.get("PATH_DATASETS", ".")
# AVAIL_GPUS = [0, 1, 2] # min(1, torch.cuda.device_count())

torch.cuda.empty_cache()

import gc; gc.collect();
from loss_fct import compute_loss_regression

from sklearn.metrics import ConfusionMatrixDisplay, cohen_kappa_score
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

cwd = os.getcwd()
path_parent = "/path/"

log_root_dir = os.path.join(path_parent, f"modeling_results")

assert os.path.exists(log_root_dir), f"Path {log_root_dir} does not exist, this is manually set here. needs to be the same as in the training script."

def ckpt_file_with_min_validation_loss(ckpt_dir):
    """ 
    from a directory with multiple .ckpt files, return the file with mimum validation loss
    """
    
    ckpts = [x for x in os.listdir(ckpt_dir) if '.ckpt' in x]
    val_losses = [float(re.search('val_loss=\d+\.?\d*', x)[0].replace('val_loss=', '')) for x in ckpts]
    file_ckpts_min_val_loss = ckpts[np.argmin(val_losses)]
    
    return file_ckpts_min_val_loss


def get_data_from_dataset(ds):
    """
    input: dataset from Dataset Class
    output: tensors of X and y
    """

    X = torch.stack([x[0] for x in ds])
    y_regression = np.array([x[1] for x in ds])
    y_classification = np.array([x[2] for x in ds])
    y_annotations = torch.stack([x[3] for x in ds])
    covariates = torch.stack([x[4] for x in ds])

    return X, y_regression, y_classification, y_annotations, covariates


def predict_routine(model, dataset, batch_size, device=None, dset_name=''):
    """
    Input: model instance, dataset (Dataset Class instance), batch size: int
    Output: y, yp
    """

    if device is None:
        device = torch.device('cuda')  # defeault cuda gpu

    model.to(device)

    no_batches = int(np.ceil(len(dataset) / batch_size))
    for i_batch in tqdm(range(no_batches), desc=f'Predicting {dset_name}', total=no_batches):

        dataset_batch = Subset(dataset, np.arange(i_batch * batch_size, (i_batch + 1) * batch_size))
        try:
            X_batch, y_regression_batch, y_classification_batch, y_annotations_batch, covariates = get_data_from_dataset(dataset_batch)
        except Exception as e:
            print(e)
            import pdb; pdb.set_trace()

        with torch.no_grad():
            yp_regression_batch, yp_classification_batch, yp_annotations_batch, features_lhl_batch = model(
                X_batch.to(device), covariates.to(device), return_features_lhl=True
            )


        if i_batch == 0:
            y_regression = np.empty((0, y_regression_batch.shape[1]))
            y_classification = np.empty((0, y_classification_batch.shape[1]))
            y_annotations = np.empty((0, y_annotations_batch.shape[1], y_annotations_batch.shape[2]))
            yp_regression = np.empty((0, y_regression_batch.shape[1]))
            yp_classification = np.empty((0, y_classification_batch.shape[1]))
            yp_annotations = np.empty((0, yp_annotations_batch.shape[1], yp_annotations_batch.shape[2]))
            features_lhl = np.empty((0, features_lhl_batch.shape[1]))


        yp_regression_batch = yp_regression_batch.detach().cpu().numpy()
        yp_classification_batch = yp_classification_batch.detach().cpu().numpy()
        yp_annotations_batch = yp_annotations_batch.detach().cpu().numpy()
        features_lhl_batch = features_lhl_batch.detach().cpu().numpy()

        # Concatenate batches
        torch.cuda.empty_cache()
        y_regression = np.concatenate([y_regression, y_regression_batch], axis=0)
        yp_regression = np.concatenate([yp_regression, yp_regression_batch], axis=0)
        y_classification = np.concatenate([y_classification, y_classification_batch], axis=0)
        yp_classification = np.concatenate([yp_classification, yp_classification_batch], axis=0)
        y_annotations = np.concatenate([y_annotations, y_annotations_batch], axis=0)
        yp_annotations = np.concatenate([yp_annotations, yp_annotations_batch], axis=0)
        features_lhl = np.concatenate([features_lhl, features_lhl_batch], axis=0)


    return y_regression, yp_regression, y_classification, yp_classification, y_annotations, yp_annotations, features_lhl
    
    
def scatter_routine_singleset(scores_true, scores_predicted, title="", savedir=None, verbose=False, cost_function_regression='rmse'):

    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    
    ax.scatter(scores_true, scores_predicted, c='k', s=8, alpha=0.5)
    ax.set_xlabel("True score")
    ax.set_ylabel("Predicted score")
    
    loss = compute_loss_regression(torch.Tensor(scores_predicted), torch.Tensor(scores_true), cost_function_regression)
    loss = np.round(loss.numpy(), 2)
    if (len(scores_true) > 2) & (len(np.unique(scores_true)) > 1):
        corr = np.round(np.corrcoef(scores_true, scores_predicted)[0, 1], 2)
        r2 = np.round(np.corrcoef(scores_true, scores_predicted)[0, 1] ** 2, 2)
    else:
        corr = np.nan
        r2 = np.nan

    ax.text(1, 0.09, f"{cost_function_regression}: {loss}", ha='right')
    ax.text(1, 0.07, f"Correlation: {corr}", ha='right')
    ax.text(1, 0.05, f" R^2       : {r2}", ha='right')

    ax.set_title(title)
    
    if verbose:
        print(f"{cost_function_regression}: {loss}")
        print(f"Correlation: {corr}")
        print(1, 0.05, f" R^2       : {r2}")

    if savedir is not None:
        title_short = 'last' if 'last' in title else title.split('sleep_oracle-epoch=')[-1] + 'epoch'
        fig.savefig(os.path.join(savedir, 'scatter_' + title_short + '.png', dpi=300))
        
    return fig


def scatter_routine_3sets(results_regression, title="", savedir=None, cost_function_regression='rmse', verbose=False):


    results_correlation ={}

    y_train = results_regression['y_regression_train']
    yp_train = results_regression['yp_regression_train']
    y_validation = results_regression['y_regression_validation']
    yp_validation = results_regression['yp_regression_validation']
    y_test = results_regression['y_regression_test']
    yp_test = results_regression['yp_regression_test']

    n_targets = y_train.shape[1]
    results_correlation['n_targets'] = n_targets

    fig, ax = plt.subplots(3, n_targets, figsize=(5, 15), sharex='col', sharey='col')

    if n_targets == 1:
        ax = ax[:, np.newaxis]

    i_axis = 0

    for i_target in range(n_targets):
        for i_axis in range(3):
            if i_axis == 0:
                subtitle = 'train'
                scores_true = y_train[:, i_target]
                scores_predicted = yp_train[:, i_target]
            elif i_axis == 1:
                subtitle = 'validation'
                scores_true = y_validation[:, i_target]
                scores_predicted = yp_validation[:, i_target]
            elif i_axis == 2:
                subtitle = 'test'
                scores_true = y_test[:, i_target]
                scores_predicted = yp_test[:, i_target]

            ax[i_axis, i_target].scatter(scores_true, scores_predicted, c='k', s=8, alpha=0.5)
            ax[i_axis, i_target].set_xlabel("True score")
            ax[i_axis, i_target].set_ylabel("Predicted score")

            loss = compute_loss_regression(torch.Tensor(scores_predicted), torch.Tensor(scores_true), cost_function_regression, target_weights_regression=[1])
            loss = loss.numpy()
            loss = np.format_float_positional(loss, precision=2)
            ax[i_axis, i_target].text(0.98, 0.15, f"{cost_function_regression}: {loss}",
                            ha='right', transform=ax[i_axis, i_target].transAxes)
            if (len(scores_true) > 2) & (len(np.unique(scores_true)) > 1):
                corr = np.round(np.corrcoef(scores_true, scores_predicted)[0, 1], 2)
                r2 = np.round(np.corrcoef(scores_true, scores_predicted)[0, 1] ** 2, 2)
            else:
                corr = np.nan
                r2 = np.nan
            ax[i_axis, i_target].text(0.98, 0.1, f"Correlation: {corr}",
                            ha='right', transform=ax[i_axis, i_target].transAxes)
            ax[i_axis, i_target].text(0.98, 0.05, f" R^2       : {r2}",
                            ha='right', transform=ax[i_axis, i_target].transAxes)

            ax[i_axis, i_target].set_title(subtitle)

            results_correlation[f'{i_target}_{subtitle}_c'] = corr

        if verbose:
            print(subtitle)
            print(f"{cost_function_regression}: {loss}")
            print(f"Correlation: {corr}")
            print(1, 0.05, f" R^2       : {r2}")

    plt.suptitle(title)
    plt.tight_layout()
    
    if savedir is not None:
        title_short = 'last' if 'last' in title else title.split('sleep_oracle-epoch=')[-1] + 'epoch'
        fig.savefig(os.path.join(savedir, 'scatter_' + title_short + '.png'), dpi=300);
        
    fig.savefig(os.path.join(log_root_dir, 'results_regression/scatter_' + title + '.png'), dpi=300);
    plt.close('all')
    del fig

    return results_correlation



def evaluate_sleep_stage_performance(y_annotations, yp_annotations, savedir, set='train', title=''):

    y_stages = y_annotations[:, :, 0]
    y_stages[y_stages == 9] = 0
    y_stages = y_stages.flatten()
    yp_stages = np.argmax(yp_annotations, axis=1).flatten()

    cohen = cohen_kappa_score(y_stages[::30], yp_stages)
    labels = ['A', 'N3', 'N2', 'N1', 'R', 'W']
    disp = ConfusionMatrixDisplay.from_predictions(y_stages[::30], yp_stages, display_labels=labels);
    fig = disp.figure_;
    fig.set_size_inches(8, 8);
    fig.savefig(os.path.join(savedir, f'cm_{set}_{title}.png'), dpi=300);
    fig.savefig(os.path.join(log_root_dir, f'results_ss_cm/cm_{set}_{title}.png'), dpi=300);

    plt.close('all')
    del fig

    return cohen


def plot_confusionmatrix(df, cmap, ax, vmin, vmax, annot=None, fmt='.2g'):

    if annot is not None: fmt='s'
    thresh = (vmin + vmax) / 2
    ax.imshow(df, interpolation='nearest', cmap=cmap, aspect="auto", vmin=vmin, vmax=vmax)
    ax.set_yticks(range(len(df)));
    ax.set_xticks(range(len(df)));
    ax.set_xticklabels(df.index, rotation=0)
    ax.set_yticklabels(df.columns, rotation=0, va='center', ha="right")
    for i, j in product(range(df.shape[0]), range(df.shape[1])):
        ax.text(j, i, str(df.iloc[i, j]).replace('-1', 'NA'), # "{:0.2f}".format(df.iloc[i, j]),
                     horizontalalignment="center", verticalalignment="center",
                     color="white" if df.iloc[i, j] < thresh else "black",
                     fontsize=8)
    # make sure subplot is square
    ax.set_aspect('equal')        

def compute_confusion_matrices(y_stages, yp_stages):

    class_names = ['N3', 'N2', 'N1', 'R', 'W']
    class_names_order = ['W', 'R', 'N1', 'N2', 'N3']
    cm = ConfusionMatrix(actual_vector=y_stages, predict_vector=yp_stages, classes=[1, 2, 3, 4, 5])
    cm = pd.DataFrame(cm.table).T
    cm.index = class_names
    cm.columns = class_names
    cm = cm.loc[class_names_order, class_names_order]
    N = cm.sum(axis=1)

    cm_norm = cm.copy()
    cm_norm = cm_norm.astype(float)
    cm_norm[:] = np.nan
    for class_name in class_names:
        cm_norm.loc[class_name, :] = np.round(cm.loc[class_name, :].div(N[class_name]) * 100, 0)
    for col in cm_norm.columns:
        for index in cm_norm.index:
            try:
                cm_norm.loc[index, col] = int(cm_norm.loc[index, col])
            except:
                continue

    cm_norm.fillna(-1, inplace=True)
    cm_norm = cm_norm.astype(int)

    return cm, cm_norm


def evaluate_sleep_stage_performance_cm_and_kappa(results_annotations):
        
    for set in ['train', 'validation', 'test']:

        y_stages = results_annotations[f'y_annotations_{set}'].astype(int)
        if y_stages.shape[1] == 39600:  # need to select every 30th epoch
            y_stages = y_stages[:, ::30, :].astype(int)

        yp_stages = results_annotations[f'yp_annotations_{set}']
        yp_stages = np.argmax(yp_stages, axis=1).astype(int)

        y_stages = y_stages.flatten()
        yp_stages = yp_stages.flatten()
        yp_stages = yp_stages[np.isin(y_stages.flatten(), [1, 2, 3, 4, 5])]
        y_stages = y_stages[np.isin(y_stages, [1, 2, 3, 4, 5])]
        assert len(yp_stages) == len(y_stages), f'Length of yp_stages_scored_only ({len(yp_stages)}) and y_stages_scored_only ({len(y_stages)}) do not match.'

        yp_stages[yp_stages == 0] = 5  # convert predicted artifact as wake
        cohen_kappa = cohen_kappa_score(y_stages, yp_stages, weights=None)
        cm, cm_norm = compute_confusion_matrices(y_stages, yp_stages)
        # save to dictionary:
        results_annotations[f'cm_absolute_{set}'] = cm
        results_annotations[f'cm_normalized_{set}'] = cm_norm
        results_annotations[f'cohen_kappa_{set}'] = cohen_kappa

    return results_annotations


def confusion_cmap():
    """Return a LinearSegmentedColormap
    """
    c = mcolors.ColorConverter().to_rgb
    seq = [(0, 0, 0), c('white')]
    seq = [(None,) * 3, 0.0] + list(seq) + [1.0, (None,) * 3]
    cdict = {'red': [], 'green': [], 'blue': []}
    for i, item in enumerate(seq):
        if isinstance(item, float):
            r1, g1, b1 = seq[i - 1]
            r2, g2, b2 = seq[i + 1]
            cdict['red'].append([item, r1, r2])
            cdict['green'].append([item, g1, g2])
            cdict['blue'].append([item, b1, b2])
    return mcolors.LinearSegmentedColormap('CustomMap', cdict)


def plot_confusion_matrices(results_annotations, title, savedir):

    # 3x2 plot: 3 sets, 2 confusion matrices (absolute and normalized)
    cmap = confusion_cmap()
    fig, axes = plt.subplots(3, 2, figsize=(6, 9))
    axes = axes.flatten()

    for j, cm_type in enumerate(['absolute', 'normalized']):
        if cm_type == 'absolute':
            title_txt = ''
        if cm_type == 'normalized':
            title_txt = '(%)'

        all_cm_vals = [results_annotations[key].values.flatten().flatten() for key in results_annotations.keys() if f'cm_{cm_type}' in key]
        all_cm_vals = np.array(all_cm_vals).flatten()
        if cm_type == 'absolute':
            # divide by 1,000:
            all_cm_vals = all_cm_vals / 1000
        vmin = np.nanpercentile(all_cm_vals[all_cm_vals>-1], 5)
        vmax = np.nanpercentile(all_cm_vals[all_cm_vals>-1], 95)

        for i, set in enumerate(['train', 'validation', 'test']):
            cm = results_annotations[f'cm_{cm_type}_{set}']
            if cm_type == 'absolute':
                if max(cm.values.flatten()) > 10000:
                    cm = np.round(cm / 1000, 0).astype(int)
            cohen_kappa = results_annotations[f'cohen_kappa_{set}']
            plot_confusionmatrix(cm, cmap=cmap, ax=axes[i*2+j], vmin=vmin, vmax=vmax, annot=None)
            axes[i*2+j].tick_params(length=0)

            # for axis on the left, add y-label
            if j == 0:
                axes[i*2+j].set_ylabel(f'{set}', rotation=90, labelpad=9)

            # for axis on the top, add title:
            if i == 0:
                axes[i*2+j].set_title(f'{cm_type} {title_txt}', pad=6)

            if j == 1:
                axes[i*2+j].text(1.2, 0.5, f"{cohen_kappa:.2f}", fontsize=11, fontweight='bold', ha='center', va='center', transform=axes[i*2+j].transAxes)
        axes[1].text(1.2, 1.05, "K", fontsize=12, ha='center', va='center', transform=axes[1].transAxes)

    plt.suptitle(f"Sleep staging - confusion matrices", x=0.5)
    plt.subplots_adjust(wspace=0.3, hspace=0.3, right=0.87)
    fig.savefig(os.path.join(savedir, f'cm_3x2_{title}.png'), dpi=300);
    fig.savefig(os.path.join(log_root_dir, f'results_ss_cm/cm_3x2_{title}.png'), dpi=300);
    fig.savefig(os.path.join(log_root_dir, f'results_ss_cm/pdf/cm_3x2_{title}.pdf'), transparent=False, facecolor='w')



def plot_regression_results(results_regression, target_var, title="", savedir=None, cost_function_regression='rmse', plot_selfsupervised=False, verbose=False):

    # remove the mean sleep feature / cog var if present:
    # target_var = [x for x in target_var if (not 'mean_sleep_features' in x) & (not 'mean_cog_features' in x)]

    target_var_plots = copy.deepcopy(target_var)

    if plot_selfsupervised:
        title += '_self-supervised'
        # if > 20 target variables, plot only 50 target variables, equally spaced:
        if len(target_var) > 20:
            target_var_plots = target_var_plots[:5] + target_var_plots[5::int(len(target_var_plots)/50)]
        fig_width = 30
    else:
        target_var_plots = [target for target in target_var_plots if not target.startswith('f_')]
        fig_width = int(3*len(target_var_plots))

    results_correlation = {}
    results_correlation['n_targets'] = len(target_var)
    
    for i_target, target in enumerate(target_var):
        for i_set, set in enumerate(['train', 'validation', 'test']):
            y = results_regression[f'y_regression_{set}'][:, i_target]
            yp = results_regression[f'yp_regression_{set}'][:, i_target]

            yp = yp[np.isfinite(y)]
            y = y[np.isfinite(y)]

            # add correlation coefficient:
            if 'dx-tm' in target:
                # do ROC AUC, roc_auc_score
                if len(np.unique(y)) > 1:
                    # compute probabilities of predicted values for the classification tasks (from logits):
                    yp = torch.sigmoid(torch.Tensor(yp)).numpy()
            
                    roc_auc = roc_auc_score(y, yp)
                    prc_auc = average_precision_score(y, yp)
                    accuracy = accuracy_score(y, yp > 0.5)
                    
                else:
                    roc_auc = np.nan
                    prc_auc = np.nan
                    accuracy = np.nan
                    
                results_correlation[f'rocauc_{target}_{set}'] = roc_auc
                results_correlation[f'prcauc_{target}_{set}'] = prc_auc
                results_correlation[f'accuracy_{target}_{set}'] = accuracy
                
            else: # do Pearson Corr:
                if (len(y) > 2) & (len(np.unique(y)) > 1):
                    # only where y is finite
                    r, p = pearsonr(y, yp)
                    
                else:
                    r, p = np.nan, np.nan
                    
                results_correlation[f'r_{target}_{set}'] = r
                results_correlation[f'p_{target}_{set}'] = p

    n_targets = len(target_var_plots)

    fig, axes = plt.subplots(3, n_targets, figsize=(fig_width, 9), sharex='col', sharey='col')
    axes = axes.flatten()

    for i_target, target in enumerate(target_var_plots):
        for i_set, set in enumerate(['train', 'validation', 'test']):
            idx_target_original = target_var.index(target)
            y = results_regression[f'y_regression_{set}'][:, idx_target_original]
            yp = results_regression[f'yp_regression_{set}'][:, idx_target_original]
            
            # only where y is finite:
            yp = yp[np.isfinite(y)]
            y = y[np.isfinite(y)]
            
            if 'dx' in target:
                # classification type of plot. Here, y only takes on two values: 0 or 1. yp is a probability value between 0 and 1.
                # add boxplots for the two classes:
                # if y is only non-finite, skip this target:
                if sum(np.isfinite(y)) == 0:
                    continue
            
                yp = torch.sigmoid(torch.Tensor(yp)).numpy()
                # x boxplot: 0 if 0, 0.5 if 1:
                x_boxplot = y
                width_boxplot = 0.8
                sns.boxplot(x=x_boxplot, y=yp, ax=axes[i_set*n_targets + i_target], width = width_boxplot,
                            palette='Set2', linewidth=1, fliersize=0)
                try:
                    sns.stripplot(x=x_boxplot, y=yp, ax=axes[i_set*n_targets + i_target], color='k', alpha=0.5, size=2)
                except:
                    pass
                
                # axes[i_set*n_targets + i_target].set_aspect('equal')
                roc_auc = results_correlation[f'rocauc_{target}_{set}']
                prc_auc = results_correlation[f'prcauc_{target}_{set}']
                try:
                    roc_auc = np.round(roc_auc, 2)
                    prc_auc = np.round(prc_auc, 2)
                except:
                    pass
                
                axes[i_set*n_targets + i_target].text(0.01, 0.9, f"AUROC: {roc_auc}\nAUPRC: {prc_auc}", fontsize=10, ha='left', va='center', transform=axes[i_set*n_targets + i_target].transAxes)
                lim_y_min = -0.02
                lim_y_max = 1.02
                lim_x_min = - width_boxplot / 2 - 0.1
                lim_x_max = 1 + width_boxplot / 2 + 0.1
                xticks = [0, 1]
                yticks = [0, 0.5, 1]
                axes[i_set*n_targets + i_target].set_xticks(xticks)
                axes[i_set*n_targets + i_target].set_xticklabels(['No', 'Yes'])

            else:
                # regression type of plot:
                performance_value = results_correlation[f'r_{target}_{set}']
                axes[i_set*n_targets + i_target].scatter(y, yp, s=8, alpha=0.5, c='k', zorder=1)
                # add linear fit:
                try:
                    slope, intercept, r_value, p_value, std_err = linregress(y, yp)
                    axes[i_set*n_targets + i_target].plot([-10, 10], np.array([-10, 10])*slope + intercept, c='r', lw=1, alpha=0.2, zorder=0)
                except:
                    pass
                axes[i_set*n_targets + i_target].plot([-10, 10], [-10, 10], c='k', lw=1, alpha=0.2, zorder=0, linestyle='--')

                # add cost function value:
                loss = compute_loss_regression(torch.Tensor(yp), torch.Tensor(y), cost_function_regression, target_weights_regression=torch.tensor([1]*len(yp))) / sum(torch.isfinite(torch.Tensor(y)))
                loss = loss.numpy()
                loss = np.format_float_positional(loss, precision=2)

                axes[i_set*n_targets + i_target].text(0.01, 0.9, f"r={performance_value:.2f}\n{cost_function_regression}: {loss}", fontsize=10, ha='left', va='center', transform=axes[i_set*n_targets + i_target].transAxes)
                lim_y_min = lim_x_min = -3
                lim_y_max = lim_x_max = 3
                xticks = [-2, 0, 2]
                yticks = [-2, 0, 2]
        
                axes[i_set*n_targets + i_target].set_aspect('equal')
                
            # for axis on the left, add text and y-label
            if i_target == 0:
                axes[i_set*n_targets + i_target].text(-0.45, 0.5, f'{set}', fontsize=12, ha='right', va='center', transform=axes[i_set*n_targets + i_target].transAxes)
                axes[i_set*n_targets + i_target].set_ylabel(f'{set}\npredicted', rotation=90, labelpad=3)

            # for axis on the bottom, add x-label
            if i_set == 2:
                axes[i_set*n_targets + i_target].set_xlabel(f'true')
            # for axis on the top, add title:
            if i_set == 0:
                axes[i_set*n_targets + i_target].set_title(f"{target.replace('domain_', '')}")
            # make axis square:
            

        axes[i_target].set_xticks(xticks)
        axes[i_target].set_yticks(yticks)
        axes[i_target].set_xlim([lim_x_min, lim_x_max])
        axes[i_target].set_ylim([lim_y_min, lim_y_max])

    if savedir is not None:
        title_short = 'last' if 'last' in title else title.split('sleep_oracle-epoch=')[-1] + 'epoch'
        fig.savefig(os.path.join(savedir, 'scatter_' + title_short + '.png'), dpi=300);
        
    # fig.savefig(os.path.join(log_root_dir, 'results_regression/scatter_' + title + '.png'), dpi=300);

    fig.suptitle(title.replace('weightSS', '\nweightSS').replace('ntr', '\nntr'), fontsize=11, y=1)  # add the run_id title to top of this pdf, mainly for the merged pdf report later.
    fig.savefig(os.path.join(log_root_dir, f'results_regression/pdf/scatter_{title}.pdf'), transparent=False, facecolor='w')
    
    plt.close('all')

    return results_correlation


def calculate_mean_features(data_plot, idx_sleep_features, idx_cog_features, idx_disease_features, results_correlation):
    mean_sleep_features = np.nanmean(data_plot[idx_sleep_features, :], axis=0)
    mean_cog_features = np.nanmean(data_plot[idx_cog_features, :], axis=0)
    mean_disease_features = np.nanmean(data_plot[idx_disease_features, :], axis=0)
    
    # and also add those data to results_correlation:
    for i_set, set_name in enumerate(['train', 'validation', 'test']):
        results_correlation[f'r_mean_sleep_features_{set_name}'] = mean_sleep_features[i_set]
        results_correlation[f'r_mean_cog_features_{set_name}'] = mean_cog_features[i_set]
        results_correlation[f'r_mean_disease_features_{set_name}'] = mean_disease_features[i_set]
        
        # set some non-used p-values, here to not break the code:
        results_correlation[f'p_mean_sleep_features_{set_name}'] = -1
        results_correlation[f'p_mean_cog_features_{set_name}'] = -1
        results_correlation[f'p_mean_disease_features_{set_name}'] = -1
        
    return results_correlation


import copy

def plot_regression_heatmap_r(results_correlation, title, target_vars):

    target_vars = copy.deepcopy(target_vars)
    results_correlation = copy.deepcopy(results_correlation)
    
    # plot the results_correlation values:
    # data_plot = np.array([results_correlation[f'{metric}_{target_var}_{set}'] for target_var in target_vars for set in ['train', 'validation', 'test'] for metric in ['r']]).reshape(len(target_vars), 3)
    # create the data_plot again but now the performance values changed, it may be "r" but can also be "roc_auc", "prc_auc", "accuracy:
    data_plot = np.zeros((len(target_vars), 3))
    for i_target, target_var in enumerate(target_vars):
        for i_set, set in enumerate(['train', 'validation', 'test']):
            if 'dx-' in target_var:
                roc_auc = results_correlation[f'rocauc_{target_var}_{set}']
                data_plot[i_target, i_set] = roc_auc
            else:
                data_plot[i_target, i_set] = results_correlation[f'r_{target_var}_{set}']


    idx_sleep_features = [i for i, target_var in enumerate(target_vars) if target_var.startswith('f_')]
    idx_cog_features = [i for i, target_var in enumerate(target_vars) if (target_var.startswith('np_')) | (target_var.startswith('cog_'))]
    idx_disease_features = [i for i, target_var in enumerate(target_vars) if target_var.startswith('dx')]
    if 1:
        mean_sleep_features = np.nanmean(data_plot[idx_sleep_features, :], axis=0)
        mean_cog_features = np.nanmean(data_plot[idx_cog_features, :], axis=0)
        mean_disease_features = np.nanmean(data_plot[idx_disease_features, :], axis=0)
        
        # and also add those data to results_correlation:
        for i_set, set_name in enumerate(['train', 'validation', 'test']):
            results_correlation[f'r_mean_sleep_features_{set_name}'] = mean_sleep_features[i_set]
            results_correlation[f'r_mean_cog_features_{set_name}'] = mean_cog_features[i_set]
            results_correlation[f'rocauc_mean_disease_features_{set_name}'] = mean_disease_features[i_set]
            
            # set some non-used p-values, here to not break the code:
            results_correlation[f'p_mean_sleep_features_{set_name}'] = -1
            results_correlation[f'p_mean_cog_features_{set_name}'] = -1
            results_correlation[f'p_mean_disease_features_{set_name}'] = -1
            
        # add those to data_plot and target_vars:
        data_plot = np.concatenate([data_plot, mean_sleep_features[np.newaxis, :], mean_cog_features[np.newaxis, :], mean_disease_features[np.newaxis, :]], axis=0)
        target_vars += ['mean_sleep_features', 'mean_cog_features', 'mean_disease_features']

    fig_height = max(0.1 * len(target_vars), 10)
    fig, ax = plt.subplots(1, 1, figsize=(4, fig_height))

    ax.imshow(data_plot, cmap='Greens', aspect='auto')
    # also print the values:
    for i_target_var, target_var in enumerate(target_vars):
        for i_set, set in enumerate(['train', 'validation', 'test']):
            # ax.text(i_set, i_target_var, np.round(results_correlation[f'r_{target_var}_{set}'], 2), ha='center', va='center', color='k', fontsize=4)
            if ('dx-' in target_var) | ('disease' in target_var):
                performance_value = np.round(results_correlation[f'rocauc_{target_var}_{set}'], 2)
            else:
                performance_value = np.round(results_correlation[f'r_{target_var}_{set}'], 2)
            ax.text(i_set, i_target_var, performance_value, ha='center', va='center', color='k', fontsize=4)
            
    ax.set_xticks(np.arange(3))
    ax.set_xticklabels(['train', 'validation','test'], fontsize=5, rotation=45)
    ax.set_yticks(np.arange(len(target_vars)))
    ax.set_yticklabels(target_vars, fontsize=5)
    plt.tight_layout()
    # ax.set_title(title)
    fig.savefig(os.path.join(log_root_dir, 'results_regression/pdf/regression_heatmap_r_' + title + '.pdf'), transparent=False, facecolor='w')


def save_results_csv(results, title, target_vars, dict_params=None):

    [results_correlation, cohen_training, cohen_validation, cohen_test] = results

    path_results_csv = os.path.join(log_root_dir, 'results_oracle.csv')
    
    cols_init = ['title']
    for target_var in target_vars:
        for set_name in ['train', 'validation', 'test']:
            if 'dx' in target_var:
                cols_init.append(f'rocauc_{target_var}_{set_name}')
                cols_init.append(f'prcauc_{target_var}_{set_name}')
            else:
                cols_init.append(f'r_{target_var}_{set_name}')
                cols_init.append(f'p_{target_var}_{set_name}')
    cols_init += ['ss_tr_cohen', 'ss_va_cohen', 'ss_te_cohen']
    cols_init += list(dict_params.keys())
    cols_to_front = ['model_name', 'path_mastertable', 'regression_targets', 'covariates', 'n_training_samples', 'weight_regression_targets', 'weight_stage', 'lr', 'lr_gamma', 'regression_pre_final_dim',
        'ss_te_cohen', 'r_age_z_test', 'r_mean_cog_features_test', 'rocauc_mean_disease_features_test', 'r_mean_sleep_features_test', 'r_cog_fluid_mgh-cog_test', 'rocauc_dx-tm-dementia_test', 'title']
    cols_init = cols_to_front + [col for col in cols_init if col not in cols_to_front]
        
    n_targets_regression = results_correlation['n_targets']
    
    df_results_tmp = pd.DataFrame(columns=cols_init)
    df_results_tmp.loc[0, 'title'] = title

    for target_var in target_vars:
        for set_name in ['train', 'validation', 'test']:
            if ('dx' in target_var) | ('disease' in target_var):
                df_results_tmp.loc[0, f'rocauc_{target_var}_{set_name}'] = results_correlation[f'rocauc_{target_var}_{set_name}']
                df_results_tmp.loc[0, f'prcauc_{target_var}_{set_name}'] = results_correlation[f'prcauc_{target_var}_{set_name}']
            else:
                df_results_tmp.loc[0, f'r_{target_var}_{set_name}'] = results_correlation[f'r_{target_var}_{set_name}']
                df_results_tmp.loc[0, f'p_{target_var}_{set_name}'] = results_correlation[f'p_{target_var}_{set_name}']
                
    df_results_tmp.loc[0, f'ss_tr_cohen'] = np.round(cohen_training, 2)
    df_results_tmp.loc[0, f'ss_va_cohen'] = np.round(cohen_validation, 2)
    df_results_tmp.loc[0, f'ss_te_cohen'] = np.round(cohen_test, 2)
    
    for key, value in dict_params.items():
        df_results_tmp.loc[0, key] = value
    
    # mean_sleep_features and mean_cog_features:
    if 1:
        for set_name in ['train', 'validation', 'test']:
            vars_results_sleep = [f'r_{f}_{set_name}' for f in target_vars if f.startswith('f_')]
            vars_results_cog = [f'r_{f}_{set_name}' for f in target_vars if (f.startswith('np_')) | (f.startswith('cog_'))]
            vars_results_disease = [f'rocauc_{f}_{set_name}' for f in target_vars if f.startswith('dx')]
            
            df_results_tmp.loc[0, f'r_mean_sleep_features_{set_name}'] = np.round(np.nanmean(df_results_tmp[vars_results_sleep]), 3)
            df_results_tmp.loc[0, f'r_mean_cog_features_{set_name}'] = np.round(np.nanmean(df_results_tmp[vars_results_cog]), 3)
            df_results_tmp.loc[0, f'rocauc_mean_disease_features_{set_name}'] = np.round(np.nanmean(df_results_tmp[vars_results_disease]), 3)
    
    # possible that not all columns are present in the dataframe, so add them:
    
    if not os.path.exists(path_results_csv):
        df_results = pd.DataFrame(columns=cols_init)
        df_results.to_csv(path_results_csv, index=False)
    
    with FileLock(path_results_csv + '.lock'):
        df_results = pd.read_csv(path_results_csv, low_memory=False)
            
        cols_to_add = [col for col in df_results.columns if col not in df_results_tmp.columns]
        df_results_tmp = pd.concat([df_results_tmp, pd.DataFrame(columns=cols_to_add)], axis=1)
        df_results_tmp = df_results_tmp[df_results.columns]
        
        for col in df_results_tmp.columns:
            try:
                if pd.isna(df_results_tmp[col]).item(): 
                    continue
            except Exception as e:
                print(e)
                import pdb; pdb.set_trace()
                blub = 1
                continue
            try:
                if (col.startswith('r_')) | (col.startswith('rocauc_')) | (col.startswith('prcauc_')):
                    df_results_tmp[col] = np.round(df_results_tmp[col].item(), 3)
                elif col.startswith('p_'):
                    df_results_tmp[col] = np.round(df_results_tmp[col].item(), 4)
            except Exception as e:
                print(e)
                import pdb; pdb.set_trace()
                blub = 1
                continue 
            
        df_results = pd.concat([df_results, df_results_tmp], axis=0)
        df_results.to_csv(path_results_csv, index=False)
        print(f"{path_results_csv}: Saved results for {title}")

### Model development tracking (visualize logger info)

def load_tensorboard_data(event_file):
    """Load tensorboard data from event file."""
    event_acc = EventAccumulator(event_file)
    event_acc.Reload()

    data = {}

    # Load scalars
    for tag in event_acc.Tags()["scalars"]:
        events = event_acc.Scalars(tag)
        data[tag] = {
            "step": [event.step for event in events],
            "value": [event.value for event in events],
        }

    return data


def create_dataframe(tensorboard_data):
    """Create a pandas dataframe from tensorboard data."""
    df = pd.DataFrame(columns=["step", "tag", "value"])

    for tag, data in tensorboard_data.items():
        temp_df = pd.DataFrame({"step": data["step"], "tag": tag, "value": data["value"]})
        df = pd.concat([df, temp_df], axis=0, ignore_index=True)

    return df


# Plot training and validation loss on the same figure
def plot_train_val_loss(ax, df, train_loss_tag, val_loss_tag, title=None):
        
    """Plot training and validation loss on the same figure."""
    if title is None:
        title = train_loss_tag.replace('_epoch', '').replace('train_', '').replace('_', ' ').capitalize()
    
    color_training = 'black'
    color_validation = 'red'

    train_loss = df[df["tag"] == train_loss_tag]
    val_loss = df[df["tag"] == val_loss_tag]

    ax.plot(train_loss["epoch"], train_loss["value"], label="Train Loss", c=color_training)
    ax.plot(val_loss["epoch"], val_loss["value"], label="Validation Loss", c=color_validation)
    # ax.set_xlabel("Epoch")
    ax.set_ylabel(title)
    # ax.set_title(title)
    ax.legend(frameon=False)

    # limit y-axis from 0 to median + IQR:
    y = np.concatenate([train_loss["value"], val_loss["value"]])
    # y = train_loss["value"]
    y = y[~np.isnan(y)]
    y = y[~np.isinf(y)]
    iqr = np.subtract(*np.percentile(y, [75, 25]))
    y_limit = np.median(y) + 2 * iqr
    if 'classification' not in train_loss_tag:
        ax.set_ylim(0, y_limit)
    # transform y to log scale
    # ax.set_yscale('log')

# Plot learning rate over time
def plot_learning_rate(ax, df, lr_tag, title="Learning Rate"):
    """Plot learning rate over time."""
    lr_data = df[df["tag"] == lr_tag]
    ax.plot(lr_data["epoch"], lr_data["value"], label="Learning Rate", c='k')
    # ax.set_xlabel("Epoch")
    ax.set_ylabel("Learning Rate")
    # ax.set_title(title)
    ax.legend(frameon=False)



def plot_train_val_loss_lr(path_eventfile, filename=None):
    """Plot training and validation loss and learning rate on the same figure."""
    if filename is None:
        filename = os.path.basename(path_eventfile)
    
    # read in tensorboard data
    tensorboard_data = load_tensorboard_data(path_eventfile)
    df = create_dataframe(tensorboard_data)
    df.sort_values(by='step', inplace=True)
    df.reset_index(inplace=True, drop=True)

    # add epoch column
    df['epoch'] = np.nan
    df.loc[df.tag == 'epoch', 'epoch'] = df.loc[df.tag == 'epoch', 'value']
    df.loc[0, 'epoch'] = 0
    df['epoch'].fillna(method='ffill', inplace=True)
    df['epoch'] = df['epoch'].astype(int)

    # plot
    fig, ax = plt.subplots(6, 1, sharex=True, figsize=(8, 8))

    i_axis = 0
    train_loss_tag = "train_loss_total_epoch"
    val_loss_tag = "val_loss_total_epoch"
    plot_train_val_loss(ax[i_axis], df, train_loss_tag, val_loss_tag)

    i_axis += 1
    train_loss_tag = "train_loss_regression_epoch"
    val_loss_tag = "val_loss_regression_epoch"
    plot_train_val_loss(ax[i_axis], df, train_loss_tag, val_loss_tag)

    i_axis += 1
    train_loss_tag = "train_loss_classification_epoch"
    val_loss_tag = "val_loss_classification_epoch"
    plot_train_val_loss(ax[i_axis], df, train_loss_tag, val_loss_tag)
    
    i_axis += 1
    train_loss_tag = "train_loss_stages_epoch"
    val_loss_tag = "val_loss_stages_epoch"
    plot_train_val_loss(ax[i_axis], df, train_loss_tag, val_loss_tag)

    i_axis += 1
    train_loss_tag = "train_cohen_stages_epoch"
    val_loss_tag = "val_cohen_stages_epoch"
    plot_train_val_loss(ax[i_axis], df, train_loss_tag, val_loss_tag)
    ax[i_axis].set_ylim([-0.1, 1])

    i_axis += 1
    lr_tag = "lr-Adam"
    plot_learning_rate(ax[i_axis], df, lr_tag)

    ax[-1].set_xlabel('Epoch')
    for i_axis in range(len(ax)):
        sns.despine(ax=ax[i_axis])
        ax[i_axis].tick_params(axis='both', length=2)

    plt.subplots_adjust(hspace=0.05)
    fig.savefig(os.path.join(os.path.dirname(path_eventfile), f'train_val_loss_lr.png'), dpi=300, bbox_inches='tight')
    fig.savefig(os.path.join(log_root_dir, f'logging_viz/{filename}.png'), dpi=300, bbox_inches='tight')

    fig.savefig(os.path.join(log_root_dir, f'logging_viz/pdf/{filename}.pdf'), transparent=False, facecolor='white', bbox_inches='tight')


def merge_pdfs(input_files, output_file):
    """
    Merge PDFs in input_files list into a single PDF file.

    Args:
    - input_files: List of input PDF file names.
    - output_file: Name of the output PDF file.
    """
    pdf_merger = PyPDF2.PdfMerger()

    for file in input_files:
        if not os.path.exists(file):
            print(f"merge_pdfs(): File {file} does not exist.")
            continue
        with open(file, 'rb') as f:
            pdf_merger.append(PyPDF2.PdfReader(f))

    with open(output_file, 'wb') as output:
        pdf_merger.write(output)


def plot_regression_results_domains(df_regression, cohort, title="", path_save=None, cost_function_regression='rmse', verbose=False):
    """
    Plot cog domain results for a specific cohort. logic: automatically get the number of cog domains, and plot them in a grid with 2 columns
    """

    if cohort == 'mgh-cog':
        target_vars = ['domain_mgh-cog_cog_fluid',
                        'domain_mgh-cog_workingmemory',
                        'domain_mgh-cog_processingspeed',
                        'domain_mgh-cog_sequencememory',
                        'domain_mgh-cog_dimensional',
                        'domain_mgh-cog_flanker']
        
    elif cohort == 'fhs':
        target_vars = ['domain_fhs_cog_fluid',
                        'domain_fhs_fingtapr_psg1_closest_z',
                        'domain_fhs_pasd_psg1_closest_z',
                        'domain_fhs_vrd_psg1_closest_z',
                        'domain_fhs_trailsb_psg1_closest_z']
        
    elif cohort == 'mesa':
        target_vars = ['domain_mesa_cog_total',
                        'domain_mesa_dsct_visit1_z',
                        'domain_mesa_casi_visit1_z',
                        'domain_mesa_dgtfor_visit1_z']
        
    target_vars = ['age_z'] + target_vars

    n_rows = int(np.ceil(len(target_vars) / 2))
    fig, axes = plt.subplots(n_rows, 2, figsize=(7, 10), sharex='col', sharey='col')
    axes = axes.flatten()

    for i_target, target in enumerate(target_vars):
        y = df_regression.query(f"cohort == '{cohort}'")[f'{target}_true'].values
        yp = df_regression.query(f"cohort == '{cohort}'")[f'{target}_pred'].values

        yp = yp[np.isfinite(y)]
        y = y[np.isfinite(y)]
        
        # plot:
        axes[i_target].scatter(y, yp, s=8, alpha=0.5, c='k', zorder=1)
        # add linear fit:
        try:
            slope, intercept, r_value, p_value, std_err = linregress(y, yp)
            axes[i_target].plot([-10, 10], np.array([-10, 10])*slope + intercept, c='r', lw=1, alpha=0.2, zorder=0)
        except:
            pass
        axes[i_target].plot([-10, 10], [-10, 10], c='k', lw=1, alpha=0.2, zorder=0, linestyle='--')


        # add correlation coefficient:
        # try:
        n_bootstrap = 1000
        if 1:
            r, p = pearsonr(y, yp)

            # bootstrap confidence interval:
            r_boot = []
            for i in range(n_bootstrap):
                idx = np.random.choice(len(y), len(y), replace=True)
                r_boot.append(pearsonr(y[idx], yp[idx])[0])
            r_boot = np.array(r_boot)
            r_boot.sort()

            # r-string with 2-decimals:            
            ci_25 = np.round(np.percentile(r_boot, 2.5), 2)
            ci_975 = np.round(np.percentile(r_boot, 97.5), 2)
            r = np.round(r, 2)
            r_string = f"r={r} ({ci_25}, {ci_975})"
            print(target, r_string)

        # except:
            # r, p = np.nan, np.nan

        if 1:
            # add cost function value:
            loss = compute_loss_regression(torch.Tensor(yp), torch.Tensor(y), cost_function_regression, target_weights_regression=torch.tensor([1]*len(yp))) / sum(torch.isfinite(torch.Tensor(y)))
            loss = loss.numpy()
            loss = np.format_float_positional(loss, precision=2)
            # as above, get 95% confidence interval:
            loss_boot = []
            for i in range(n_bootstrap):
                idx = np.random.choice(len(y), len(y), replace=True)
                loss_boot.append(compute_loss_regression(torch.Tensor(yp[idx]), torch.Tensor(y[idx]), cost_function_regression, target_weights_regression=torch.tensor([1]*len(yp[idx]))) / sum(torch.isfinite(torch.Tensor(y[idx]))))
            loss_boot = np.array(loss_boot)
            loss_boot.sort()
            ci_25 = np.round(np.percentile(loss_boot, 2.5), 2)
            ci_975 = np.round(np.percentile(loss_boot, 97.5), 2)
            # loss = np.round(loss, 2)
            loss_string = f"{loss} ({ci_25}, {ci_975})"
            print(target, f"{cost_function_regression}={loss_string}")

        axes[i_target].text(0.01, 0.9, f"{r_string}\n{cost_function_regression}: {loss_string}", fontsize=9, ha='left', va='center', transform=axes[i_target].transAxes)
            
        axes[i_target].set_aspect('equal')
        lim_y_min = -3
        lim_y_max = 3
        axes[i_target].set_xlim([lim_y_min, lim_y_max])
        axes[ i_target].set_ylim([lim_y_min, lim_y_max])
        axes[i_target].set_xticks([-2, 0, 2])
        axes[i_target].set_yticks([-2, 0, 2])

        # for left column:
        if i_target % 2 == 0:
            axes[i_target].set_ylabel(f'predicted', rotation=90, labelpad=3)

        # for last row:
        if i_target >= len(target_vars) - 2:
            axes[i_target].set_xlabel(f'true')

        axes[i_target].set_title(target.replace('domain_', '').replace(f'{cohort}_', ''), pad=2)

    # if last subplot is not used, remove it:
    if i_target == len(axes) - 2:
        fig.delaxes(axes[i_target+1])
        # and then, also add 'true' to the second-last subplot:
        axes[i_target-1].set_xticks([-2, 0, 2])
        axes[i_target-1].set_xlabel(f'true')


    fig.suptitle(f'{title}', y=1, fontweight='bold')
    plt.subplots_adjust(top=0.95, wspace=0.12, hspace=0.15)
    
    if path_save is not None:
        fig.savefig(path_save + '.png', dpi=300);
        
