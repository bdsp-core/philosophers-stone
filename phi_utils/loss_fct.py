import torch
torch.manual_seed(42)
import random
# random.seed(42)
import numpy as np
np.random.seed(42)
import pandas as pd
from torch.nn import Linear, ReLU, CrossEntropyLoss, Sequential, Conv2d, MaxPool2d, Module, Softmax, BatchNorm2d, Dropout, CosineSimilarity, HuberLoss
import torch.nn.functional as F
from sklearn.metrics import cohen_kappa_score

# functions used by all models:

def calculate_loss_MLT(self, batch):

    verbose = False
    cost_function_name = self.hparams.cost_function
    target_weights_regression = self.hparams.target_weights_regression
    target_weights_classification = self.hparams.target_weights_classification
    x, y_regression, y_classification, y_annotations, cohort_encoding = batch

    y_regression = y_regression.float().squeeze()

    yp_regression, yp_classification, yp_stages = self(x, cohort_encoding)
    yp_regression = yp_regression.float().squeeze()
    yp_stages = yp_stages.float()

    loss_regression = compute_loss_regression(yp_regression, y_regression, cost_function_name, target_weights_regression)

    loss_classification = compute_loss_classification(yp_classification, y_classification, target_weights_classification)
    
    y_stages = y_annotations[:, :, 0]
    loss_stages, cohen_kappa = compute_loss_stages(yp_stages, y_stages)

    if verbose:
        print(f'loss regression {loss_regression:.3f}', f'loss stages {loss_stages:.3f}', f'loss classification {loss_classification:.3f}')

    cf_weight_classification = self.hparams.cf_weight_classification
    cf_weight_stage = self.hparams.cf_weight_stage
    cf_weight_regression = 1 - cf_weight_stage - cf_weight_classification

    loss_lhl = cf_weight_regression * loss_regression + cf_weight_classification * loss_classification # "LHL" = "Last Hidden Layer"
    loss_total = cf_weight_regression * loss_regression + cf_weight_stage * loss_stages + cf_weight_classification * loss_classification
    if verbose:
        print('cf_weight_regression', cf_weight_regression, 'cf_weight_classification', cf_weight_classification, 'cf_weight_stage', cf_weight_stage)
        print('loss_regression', loss_regression, 'loss_classification', loss_classification, 'loss_stages', loss_stages)
        print(f'loss_total {loss_total}')
        
    return loss_total, loss_regression, loss_classification, loss_lhl, loss_stages, cohen_kappa


def compute_loss_classification(yp_classification, y_classification, target_weights_classification):
    
    device = yp_classification.device  # Get the device of yp_classification
    loss_classification = torch.tensor(0.0, device=device)  # Create the tensor on the same device

    n_valid_targets_in_batch = 0
    # loop over all classification tasks
    for i in range(y_classification.shape[1]):
      # mask non-finite y values: [y-value can be nan if no value is available for a sample (i.e. a disease category is not available for a subject)]
        mask = torch.isfinite(y_classification[:, i])
        yp_classification_i = yp_classification[mask][:, i]
        y_classification_i = y_classification[mask][:, i]
        
        if y_classification_i.nelement() == 0:
            # No valid target values for this classification task in this batch
            continue
        
        n_valid_targets_in_batch += 1
        
        yp_classification_i = torch.stack([1 - yp_classification_i, yp_classification_i], dim=1)
        y_classification_i = y_classification_i.long()
        
        loss_classification_tmp = CrossEntropyLoss()(yp_classification_i, y_classification_i)
        loss_classification += loss_classification_tmp * target_weights_classification[i]

    # normalize loss by number of valid targets in batch
    if n_valid_targets_in_batch > 0:
        loss_classification /= n_valid_targets_in_batch

    return loss_classification
        
def calculate_loss(self, batch):

    verbose = False
    cost_function_name = self.hparams.cost_function
    target_weights_regression = self.hparams.target_weights_regression
    x, y = batch
    y = y.float().squeeze()
    x = self(x).float().squeeze()

    loss = compute_loss_regression(x, y, cost_function_name, target_weights_regression)

    return loss

def compute_loss_stages(yp_stages, y_stages):

    # auxiliary tasks such as sleep staging, respiratory event detection, these tasks
    # use an array of y, pixel-wise cost function.
    # print('ystages pre-resample', y_stages.shape, 'yp_stages', yp_stages.shape)
    verbose = False
    
    if y_stages.shape[1] == 1320: # EEG timeseries version: stage saved in 30-second epoch resolution
        # these models' decoder outputs a shape of (batch_size, 6, 1320) [1320 30-second epochs in 11 hours]
        pass
        
    else:  # all the spectrogram versions
        # these models' decoder outputs a shape of (batch_size, 6, 1320) [1320 30-second epochs in 11 hours]
        # y_stages is passed here in 1Hz resolution, resample to 30-second-epoch resolution.
        # Todo/future: resample the saved data for efficiency.
        y_stages = y_stages[:, ::30]
        
    n_classes = 6 # 5 physiological [1-5], 1 "artefact/not physiological/padding [0]"
    # print('ystages post-resample', y_stages.shape, 'yp_stages', yp_stages.shape)

    # compute weights for stage imbalance, per batch.
    counts = y_stages.unique(return_counts=True, sorted=True)[1]

    if len(counts) < n_classes: # not all classes in batch, just use weights 1 here
        weight = torch.ones(n_classes,).to(y_stages.device)
    else:
        weight = 1 - counts / sum(counts)

    assert len(weight) == n_classes, f"Stage weight length {len(weight)} != expected {n_classes}"

    loss_fct_ce = CrossEntropyLoss(weight=weight)

    loss_ce = loss_fct_ce(yp_stages, y_stages)
    loss_dice = dice_loss(F.softmax(yp_stages, dim=1).float(),
                      F.one_hot(y_stages, n_classes).to(torch.float32).float().permute(0, 2, 1),
                      multiclass=True)
    
    # compute additional metrics
    # Cohen's Kappa
    if 1:
        yp_stages_scored_only = yp_stages.detach().cpu().numpy()
        y_stages_scored_only = y_stages.detach().cpu().numpy()
        yp_stages_scored_only = np.argmax(yp_stages_scored_only, axis=1)
        yp_stages_scored_only = yp_stages_scored_only.flatten()
        y_stages_scored_only = y_stages_scored_only.flatten()
        yp_stages_scored_only = yp_stages_scored_only[np.isin(y_stages_scored_only.flatten(), [1, 2, 3, 4, 5])]
        y_stages_scored_only = y_stages_scored_only[np.isin(y_stages_scored_only, [1, 2, 3, 4, 5])]
        assert len(yp_stages_scored_only) == len(y_stages_scored_only), f'Length of yp_stages_scored_only ({len(yp_stages_scored_only)}) and y_stages_scored_only ({len(y_stages_scored_only)}) do not match.'
        yp_stages_scored_only[yp_stages_scored_only == 0] = 5  # set predicted artifact as wake here
        cohen_kappa = cohen_kappa_score(y_stages_scored_only, yp_stages_scored_only, weights=None)
    else:
        cohen_kappa = 0

    if verbose:
        print('                                           ', f'ce:   {loss_ce:.3f}', f'dice: {loss_dice:.3f}, cohen_kappa: {cohen_kappa:.3f}')

    loss = (loss_ce + loss_dice) / 2

    return loss, cohen_kappa


def compute_loss_regression(x, y, cost_function_name, target_weights_regression=None, y_annotations=None):
    verbose=False
    """x: predicted, y: true"""
    if cost_function_name == 'mae':
        loss_fct = torch.nn.L1Loss()
        loss = loss_fct(x, y)
    elif cost_function_name == 'rmse':

        # mask non-finite y values: [y-value can be nan if no value is available for a sample (i.e. a cognitive score is not available for a subject)]
        mask = torch.isfinite(y)
        x = x[mask]
        y = y[mask]
        target_weights_regression = target_weights_regression[mask]

        if x.nelement() == 0:
            return torch.tensor(0.0)

        loss_fct = torch.nn.MSELoss(reduction='none')  # return elementwise squared error
        loss = loss_fct(x, y)
        if not torch.isfinite(x).all():
            print('x contains nan or inf values.')
        
        if loss.dim() > 1: 
            loss = torch.sqrt(torch.nanmean(loss, dim=0))  # RMSE for each target

        if loss.dim() > 0:
            # weighted sum of loss elements (weighted by target_weights_regression):
            loss = torch.sum(loss * target_weights_regression) # weighted sum
            
    elif 'huber' in cost_function_name:
        
        # mask non-finite y values: [y-value can be nan if no value is available for a sample (i.e. a cognitive score is not available for a subject)]
        mask = torch.isfinite(y)
        x = x[mask]
        y = y[mask]
        target_weights_regression = target_weights_regression[mask]

        if x.nelement() == 0:
            return torch.tensor(0.0)

        delta = 1  # default
        if cost_function_name == 'huber1':
            delta = 1
        elif cost_function_name == 'huber2':
            delta = 2
        elif cost_function_name == 'huber3':
            delta = 3
            
        loss_fct = HuberLoss(reduction='none', delta=delta)  # return elementwise error
        loss = 2 * loss_fct(x, y)  # scale by 2 to match MSE scale
        
        if loss.dim() > 0:
            # weighted sum of loss elements (weighted by target_weights_regression):
            loss = torch.sum(loss * target_weights_regression) # weighted sum

    elif cost_function_name == 'rmse_pearson':
        loss_fct_mse = torch.nn.MSELoss()
        loss_mse = loss_fct_mse(x, y) # mse
        loss_mse = torch.sqrt(loss_mse) # rmse
        loss_mse = loss_mse / ((y.max() - y.min()).abs() + 1) # rough scaling for now
        cos = CosineSimilarity(dim=0, eps=1e-6)
        pearson = cos(x - x.mean(), y - y.mean())
        if verbose:
            print(f'p={pearson},   LOSS:1-p ={1-pearson}, mse={loss_mse}')
        loss = (1 - pearson) + loss_mse
    elif cost_function_name == 'mae_pearson':
        loss_fct_mae = torch.nn.L1Loss()
        loss_mae = loss_fct_mae(x, y)
        loss_mae = loss_mae / ((y.max() - y.min()).abs() + 1) # rough scaling for now
        cos = CosineSimilarity(dim=0, eps=1e-6)
        pearson = cos(x - x.mean(), y - y.mean())
        if verbose:
            print(f'p={pearson},   LOSS:1-p ={1-pearson}, mae={loss_mae}')
        loss = (1 - pearson) + loss_mae
    else:
        raise ValueError('Cost function not expected/implemented. Needs to be added.')

    return loss



import torch
from torch import Tensor

def dice_coeff(input: Tensor, target: Tensor, reduce_batch_first: bool = False, epsilon=1e-6):
    # Average of Dice coefficient for all batches, or for a single mask
    assert input.size() == target.size()
    if input.dim() == 1 and reduce_batch_first:
        raise ValueError(f'Dice: asked to reduce batch but got tensor without batch dimension (shape {input.shape})')

    if input.dim() == 1 or reduce_batch_first:
        inter = torch.dot(input.reshape(-1), target.reshape(-1))
        sets_sum = torch.sum(input) + torch.sum(target)
        if sets_sum.item() == 0:
            sets_sum = 2 * inter

        return (2 * inter + epsilon) / (sets_sum + epsilon)
    else:
        # compute and average metric for each batch element
        dice = 0
        for i in range(input.shape[0]):
            dice += dice_coeff(input[i, ...], target[i, ...])
        return dice / input.shape[0]


def multiclass_dice_coeff(input: Tensor, target: Tensor, reduce_batch_first: bool = False, epsilon=1e-6):
    # Average of Dice coefficient for all classes
    assert input.size() == target.size()
    dice = 0
    for channel in range(input.shape[1]):
        dice += dice_coeff(input[:, channel, ...], target[:, channel, ...], reduce_batch_first, epsilon)

    return dice / input.shape[1]


def dice_loss(input: Tensor, target: Tensor, multiclass: bool = False):
    # Dice loss (objective to minimize) between 0 and 1
    assert input.size() == target.size()
    fn = multiclass_dice_coeff if multiclass else dice_coeff
    return 1 - fn(input, target, reduce_batch_first=True)
