import sys
import os

base_path1 = os.path.join( os.path.dirname(__file__), '..', 'pytorch-image-models')
base_path2 = os.path.join( os.path.dirname(__file__), '..', 'pytorch-image-models/timm/')
sys.path.insert(0, os.path.abspath(base_path1))
sys.path.insert(0, os.path.abspath(base_path2))

from models.maxxvit import *
from models.maxxvit import _rw_max_cfg, cfg_window_size, MaxxVitStage, _init_conv
from models._manipulate import named_apply
from layers import get_norm_layer, get_norm_act_layer, to_2tuple
from layers import NormMlpClassifierHead2
from torch import nn

from typing import Union, Tuple
from functools import partial
import warnings
warnings.filterwarnings("ignore", message="Initializing zero-element tensors is a no-op")
warnings.filterwarnings("ignore", message="torch.meshgrid: in an upcoming release, it will be required to pass the indexing argument.")

import torch
import random
import numpy as np
from pytorch_lightning import LightningModule
from torch.nn import Module, ReLU, Sequential, Conv2d, ConvTranspose2d, ConstantPad2d
import torch.optim as optim
from phi_utils.loss_fct import calculate_loss_MLT
from torch.optim.lr_scheduler import LambdaLR


torch.manual_seed(42)
# random.seed(67)
np.random.seed(42)

class Reshape(Module):
    def __init__(self, *args):
        super(Reshape, self).__init__()
        self.shape = args

    def forward(self, x):
        return x.view(x.size(0), *self.shape)


class SliceLastDim(Module):
    def __init__(self, n_remove):
        super(SliceLastDim, self).__init__()
        self.n_remove = n_remove

    def forward(self, x):
        return x[:, :, :, :-self.n_remove]

class LinearWarmupLinearDecayWithRestartLR(LambdaLR):
    def __init__(self, optimizer, warmup_steps, total_steps, restart_step, min_lr_factor=0.1, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.restart_step = restart_step
        self.min_lr_factor = min_lr_factor
        super().__init__(optimizer, self.lr_lambda, last_epoch)

    def lr_lambda(self, step):
        if step < self.warmup_steps:
            return float(step) / float(max(1, self.warmup_steps))
        elif step < self.restart_step:
            return max(self.min_lr_factor, 
                    float(self.restart_step - step) / float(max(1, self.restart_step - self.warmup_steps)))
        elif step == self.restart_step:
            return 0.9  # Reset LR to 90% of initial value
        else:
            return max(self.min_lr_factor,
                        float(self.total_steps - step) / float(max(1, self.total_steps - self.restart_step)))


def configure_optimizers_(self):

    # optimizer = optim.Adam(self.parameters(), lr=self.hparams.lr)
    optimizer = optim.AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.lr_gamma)

    stepsize = 1000 # in epochs or batches, depending on what is set in configure_optimizers(). i had 2 for epochs, now do 200 for batches.
    
        # Calculate total steps
    total_steps = self.trainer.estimated_stepping_batches
    warmup_steps = int(min(0.05 * total_steps, 1000))  # 5% of total steps for warm-up
    restart_step = int(min(0.15 * total_steps, 10000))  # Restart at 15% of total steps
    
    print(f"\n\n\ntotal_steps: {total_steps}, \
        warmup_steps: {warmup_steps}, restart_step: {restart_step}\n\n\n")
    
    lr_scheduler = LinearWarmupLinearDecayWithRestartLR(
        optimizer,
        warmup_steps=warmup_steps,  # 10% of total steps for warm-up
        total_steps=total_steps,
        restart_step=restart_step, # Restart at 30% of total steps
        min_lr_factor=0.1,  # Minimum learning rate will be 10% of initial lr
    )
    return [optimizer], [lr_scheduler]


class SleepPhilosophersStone(LightningModule):
    """
    Base class for Philosopher Stone models. Implements the training, validation and test steps,
    configure optimizer and loss function.
    Child classes need to implement the init with model architecture and forward pass.
    """

    def __init__(self, lr, cost_function, n_targets_regression, target_weights_regression, n_targets_classification,
                 cf_weight_classification, target_weights_classification, cf_weight_stage, ntr, lr_gamma):
        
        super().__init__()
        self.save_hyperparameters()

    def configure_optimizers(self):
        [optimizer], [lr_scheduler] = configure_optimizers_(self)
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': lr_scheduler,
                'interval': 'step',  # 'step' updates after optimizer step, 'epoch' after each epoch (default)
                }
        }

    def training_step(self, batch, batch_idx):
        loss_total, loss_regression, loss_classification, loss_lhl, loss_stages, cohen_kappa_stages = self._calculate_loss(
            batch, mode="train")
        self.log_dict({"train_loss_total": loss_total,
                    "train_loss_regression": loss_regression,
                    "train_loss_classification": loss_classification,
                    "train_loss_lhl": loss_lhl,
                    "train_loss_stages": loss_stages,
                    'train_cohen_stages': cohen_kappa_stages},
                      on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        return loss_total

    def validation_step(self, batch, batch_idx):
        loss_total, loss_regression, loss_classification, loss_lhl, loss_stages, cohen_kappa_stages = self._calculate_loss(
            batch, mode="val")
        self.log_dict({"val_loss_total": loss_total,
                       "val_loss_regression": loss_regression,
                       "val_loss_classification": loss_classification,
                       "val_loss_lhl": loss_lhl,
                       "val_loss_stages": loss_stages,
                       'val_cohen_stages': cohen_kappa_stages}, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        return loss_total

    def test_step(self, batch, batch_idx):
        loss_total, loss_regression, loss_classification, loss_lhl, loss_stages, cohen_kappa_stages = self._calculate_loss(
            batch, mode="test")
        self.log_dict({"test_loss_total": loss_total,
                       "test_loss_regression": loss_regression,
                        "test_loss_classification": loss_classification,
                        "test_loss_lhl": loss_lhl,
                       "test_loss_stages": loss_stages,
                       'test_cohen_stages': cohen_kappa_stages}, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        return loss_total

    def _calculate_loss(self, batch, mode="train"):
        loss_total, loss_regression, loss_classification, loss_lhl, loss_stages, cohen_kappa_stages = calculate_loss_MLT(
            self, batch)
        return loss_total, loss_regression, loss_classification, loss_lhl, loss_stages, cohen_kappa_stages



embed_dim = 128

class Stem(nn.Module):

    def __init__(
            self,
            in_chs: int,
            out_chs: int,
            kernel_size: int = 3,
            padding: str = '',
            bias: bool = False,
            act_layer: str = 'gelu',
            norm_layer: str = 'batchnorm2d',
            norm_eps: float = 1e-5,
            fs_time: float = 1,
    ):
        super().__init__()
        if not isinstance(out_chs, (list, tuple)):
            out_chs = to_2tuple(out_chs)

        norm_act_layer = partial(get_norm_act_layer(norm_layer, act_layer), eps=norm_eps)
        self.out_chs = out_chs[-1]

        # Conditional initialization based on fs_time
        if fs_time == 1:
            
            self.conv1 = Conv2d(1, 16, kernel_size=(3, 2), stride=(2, 1))
            
        elif fs_time == 2:
            
            self.conv_pre1 = Conv2d(1, 16, kernel_size=(3, 2), stride=(2, 1), padding=(1, 0))
            self.norm_pre1 = norm_act_layer(16)
            self.padding_pre1 = ConstantPad2d((0, 1, 0, 0), 0)
            
            self.conv1 = Conv2d(16, 16, kernel_size=(3, 2), stride=(2, 1))
            
        elif fs_time == 4:

           self.conv_pre1 = Conv2d(1, 16, kernel_size=(3, 2), stride=(2, 1), padding=(1, 0))
           self.norm_pre1 = norm_act_layer(16)
           self.padding_pre1 = ConstantPad2d((0, 1, 0, 0), 0)
          
           self.conv_pre2 = Conv2d(16, 16, kernel_size=(3, 2), stride=(2, 1), padding=(1, 0))
           self.norm_pre2 = norm_act_layer(16)
           self.padding_pre2 = ConstantPad2d((0, 1, 0, 0), 0)
          
           self.conv1 = Conv2d(16, 16, kernel_size=(3, 2), stride=(2, 1))
        
        elif fs_time == 8:
            
            self.conv_pre1 = Conv2d(1, 16, kernel_size=(3, 2), stride=(2, 1), padding=(1, 0))
            self.norm_pre1 = norm_act_layer(16)
            self.padding_pre1 = ConstantPad2d((0, 1, 0, 0), 0)
            
            self.conv_pre2 = Conv2d(16, 16, kernel_size=(3, 2), stride=(2, 1), padding=(1, 0))
            self.norm_pre2 = norm_act_layer(16)
            self.padding_pre2 = ConstantPad2d((0, 1, 0, 0), 0)
            
            self.conv_pre3 = Conv2d(16, 16, kernel_size=(3, 2), stride=(2, 1), padding=(1, 0))
            self.norm_pre3 = norm_act_layer(16)
            self.padding_pre3 = ConstantPad2d((0, 1, 0, 0), 0)
            
            self.conv1 = Conv2d(16, 16, kernel_size=(3, 2), stride=(2, 1))
            
        else:
            raise ValueError(f'fs_time {fs_time} not supported')

        self.norm1 = norm_act_layer(16)
        self.conv2 = Conv2d(16, 32, kernel_size=(3, 2), stride=(2, 1))
        self.norm2 = norm_act_layer(32)
        self.conv3 = Conv2d(32, 64, kernel_size=(3, 2), stride=(2, 1))
        self.norm3 = norm_act_layer(64)
        self.conv4 = Conv2d(64, embed_dim, kernel_size=(3, 2), stride=(2, 2))
        self.pad_end = ConstantPad2d((0, 0, 0, 1), 0)

    def init_weights(self, scheme=''):
        named_apply(partial(_init_conv, scheme=scheme), self)

    def forward(self, x):
        
        verbose = False
        if verbose:
            print('Stem input', x.shape)
        
        if hasattr(self, 'conv_pre1'):
            x0 = self.conv_pre1(x)
            x0 = self.norm_pre1(x0)
            x0 = self.padding_pre1(x0)
            x = x0
            if verbose:
                print('Stem post conv_pre1', x.shape)
                
        if hasattr(self, 'conv_pre2'):
            x0 = self.conv_pre2(x)
            x0 = self.norm_pre2(x0)
            x0 = self.padding_pre2(x0)
            x = x0
            if verbose:
                print('Stem post conv_pre2', x.shape)
                
        if hasattr(self, 'conv_pre3'):
            x0 = self.conv_pre3(x)
            x0 = self.norm_pre3(x0)
            x0 = self.padding_pre3(x0)
            x = x0
            if verbose:
                print('Stem post conv_pre3', x.shape)
            
        x1 = self.conv1(x)
        x1 = self.norm1(x1)
        if verbose:
            print('Stem post conv1', x1.shape)
        x2 = self.conv2(x1)
        x2 = self.norm2(x2)
        if verbose:
            print('Stem post conv2', x2.shape)
        x3 = self.conv3(x2)
        x3 = self.norm3(x3)
        if verbose:
            print('Stem post conv3', x3.shape)
        x4 = self.conv4(x3)
        x4 = x4[:, :, :2472, :]
        # remove 2 elements, 2472 has better divisors for window/grid partition later on, and is a better 2^n
        if verbose:
            print('Stem post conv4', x4.shape)
        return x1, x2, x3, x4
    
    
class SliceModule(nn.Module):
    def __init__(self, length):
        super(SliceModule, self).__init__()
        self.length = length

    def forward(self, x):
        return x[:, :, :self.length, :]
    
class MaxVit_SleepStagesDecoder(Module):
    def __init__(self, n_chs, cfg, transformer_cfg, feat_size,
                 act_layer: str = 'gelu',
                norm_layer: str = 'batchnorm2d',
                norm_eps: float = 1e-5,
                fs_time: float = 1,
                ):
        
        super(MaxVit_SleepStagesDecoder, self).__init__()

        self.layer1_maxvit = Sequential(
            MaxxVitStage(
                n_chs,
                n_chs,
                depth=1,
                block_types='M',
                conv_cfg=cfg.conv_cfg,
                transformer_cfg=transformer_cfg,
                feat_size=feat_size,
                stride = 1,
                drop_path=[0],
            )
        )
        norm_act_layer = partial(get_norm_act_layer(norm_layer, act_layer), eps=norm_eps)

        self.layer1 = Sequential(
            # Add decoder layers here (usually ConvTranspose2d layers)
            ConvTranspose2d(n_chs, 512, kernel_size=(3, 2), stride=(2, 2)),
            ReLU(inplace=True),
            ConstantPad2d((0, 0, 0, 0), 0),
            ConvTranspose2d(512, 256, kernel_size=(3, 2), stride=(2, 2)),
            ReLU(inplace=True),
            ConstantPad2d((0, 0, 0, 1), 0),
            ConvTranspose2d(256, 128, kernel_size=(3, 2), stride=(2, 2)),
            ReLU(inplace=True),
            ConstantPad2d((0, 0, 0, 1), 0),
            ConvTranspose2d(128, 64, kernel_size=(3, 2), stride=(2, 2)),
            ReLU(inplace=True),
            ConstantPad2d((0, 1, 0, 0), 0),
        )
        self.layer2 = Sequential(
            ConvTranspose2d(64 * 2, 32, kernel_size=(3, 2), stride=(2, 1)),
            ReLU(inplace=True),
            ConstantPad2d((0, 0, 0, 0), 0),
        )
        self.layer3 = Sequential(
            ConvTranspose2d(32 * 2, 16, kernel_size=(3, 2), stride=(2, 1)),
            ReLU(inplace=True),
            ConstantPad2d((0, 0, 0, 0), 0),
        )
        self.layer4 = Sequential(
            ConvTranspose2d(16 * 2, 1, kernel_size=(3, 2), stride=(2, 1)),
            ReLU(inplace=True),
            ConstantPad2d((0, 0, 0, 1), 0),
        )
        # one scalaer per 30 seconds data, use Conv2d with (30 * fs_time, 100) kernel and stride 30 seconds.
        if 30 % fs_time == 0:
            self.layer_final = Sequential(
                Conv2d(1, 6, kernel_size=(int(30 // fs_time), 100), stride=int(30 // fs_time)),
            )
        else: # additionally, need to do some slicing to get the correct output size (throw away the last small part).
            self.layer_final = Sequential(
                Conv2d(1, 6, kernel_size=(int(30 // fs_time), 100), stride=int(30 // fs_time)),
                # Slice the result to 1320 * fs_time: (39600÷30=1320)
                SliceModule(int(1320 * fs_time))
            )
            
    def forward(self, x, skip_outputs):
        verbose = False
        
        if verbose:
            print('MaxVit_SleepStagesDecoder pre layer1', x.shape)
        x = self.layer1_maxvit(x)
        # remove last element of x: 2, 64, 4951, 97 -> 2, 64, 4950, 97
        x = x[:, :, :-1, :]
        if verbose:
            print('MaxVit_SleepStagesDecoder post layer1', x.shape)
        x = self.layer1(x)
        if verbose:
            print('MaxVit_SleepStagesDecoder -1', x.shape, skip_outputs[-1].shape)
        # Combine with the corresponding skip connection
        x = torch.cat((x, skip_outputs[-1]), 1)
        x = self.layer2(x)
        if verbose:
            print('MaxVit_SleepStagesDecoder -2', x.shape, skip_outputs[-2].shape)
        # Combine with the corresponding skip connection
        x = torch.cat((x, skip_outputs[-2]), 1)
        x = self.layer3(x)
        if verbose:
            print('MaxVit_SleepStagesDecoder -3', x.shape, skip_outputs[-3].shape)
        # Combine with the corresponding skip connection
        x = torch.cat((x, skip_outputs[-3]), 1)
        x = self.layer4(x)
        if verbose:
            print('MaxVit_SleepStagesDecoder pre final', x.shape) # pre final torch.Size([1, 1, 39600, 100]) for any fs_time
        x = self.layer_final(x)
        if verbose:
            print('MaxVit_SleepStagesDecoder final pre squeeze', x.shape) # final pre squeeze torch.Size([1, 6, 1320 * FS_TIME, 1])
        x = x.squeeze(dim=3) # (B, #classes, 1320, 1) --> (B, #classes, 1320)
        if verbose:
            print('MaxVit_SleepStagesDecoder final', x.shape)
        return x

class Print(Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        print('PrintClass', x.shape)
        return x


class SleepPhilosopherSpectral(SleepPhilosophersStone):
    """ 
    Main class for the Sleep Philosophers Stone model.
    This class implements the model architecture and the forward pass.
    It is a subclass of SleepPhilosophersStone module, which provides the basic functionality for all PhilosopherStone modules tested.
    The model is based on a mix of the MaxxVit architecture, CNNs and U-Net, which is a transformer-based model for sleep stage classification.
    The input to the model is a sleep EEG spectrogram, (39600, 100) for 11 hours of data with timestep 1 second and 100 frequency bins.
    """

    def __init__(
            self,
            lr, cost_function, n_targets_regression, target_weights_regression, n_targets_classification,
            cf_weight_classification, target_weights_classification, cf_weight_stage, ntr, lr_gamma, n_covariates,
            target_names, target_output_dims, task_types,
            fs_time = 1, # sampling frequency in seconds of the input image. The first layer(s) of the CNN stem are adjusted based on the fs_time.
            img_size: Union[int, Tuple[int, int]] = None,
            in_chans: int = 1,
            global_pool: str = 'avg',
            drop_rate: float = 0.,
            drop_path_rate: float = 0.,
            dim_attention_head: int = 64,  # number of attention heads = embeddeding_dim // dim_attention_head
            dim_final_latent_space=1024,
            **kwargs,
    ):
        if img_size is None:
            img_size = (int(39600 * fs_time), 100)
            
        super().__init__(lr, cost_function, n_targets_regression, target_weights_regression, 
                            n_targets_classification, cf_weight_classification, target_weights_classification, cf_weight_stage, ntr, lr_gamma)
        
        self.save_hyperparameters()
        
        # embed_dim represents the number of channels in each stage...
        # ... which in transformer language is the dimension of every token.
        # number of tokens: given by HxW in each stage of the model. linear embedding dimension of token=embed_dim.
        # depths represents the number of blocks in each stage. each stage may have different hidden/feature size and number of blocks.
        # block_type represents the type of block in each stage.
        # head hidden size: fully connected layer hidden layer size (1 hidden layer).

        cfg = MaxxVitCfg(
            embed_dim=(256, 512, 1024),       # dimension of every token in each stage, or in other language: number of channels in each stage
            depths=(1, 1, 1),              # number of blocks in each stage  
            block_type=('M', 'M', 'M'),      # block type for each stage
            stem_width=(16, 32, 64, embed_dim),        # CNN STEM
            head_hidden_size=768,        # MLP hidden layer size
            **_rw_max_cfg(
                dim_head=dim_attention_head, # dimension of attention heads (keys, queries, values)
                rel_pos_type='mlp',
                ),
            )
        ### dimension of attention heads (keys, queries, values) is embed_dim, set to 32 by default.
        ### number of attention heads per stage is embed_dim // attn_head_dim.

        img_size = to_2tuple(img_size)
        if kwargs:
            cfg = _overlay_kwargs(cfg, **kwargs)

        transformer_cfg = cfg_window_size(cfg.transformer_cfg, img_size)
        print_cfg = False
        if print_cfg:
            print(f'cfg.transformer_cfg.window_size', cfg.transformer_cfg.window_size
                , f'cfg.transformer_cfg.grid_size', cfg.transformer_cfg.grid_size)
            print(f'cfg.transformer_cfg.window_size', cfg.transformer_cfg.window_size)

        self.n_targets_regression = n_targets_regression
        self.n_targets_classification = n_targets_classification
        self.global_pool = global_pool
        self.embed_dim = cfg.embed_dim[-1] 
        self.num_features = cfg.embed_dim[-1]
        self.drop_rate = drop_rate
        self.grad_checkpointing = False
        self.feature_info = []
        self.dim_final_latent_space = dim_final_latent_space
        self.target_names = target_names
        self.target_output_dims = target_output_dims
        self.task_types = task_types
        
        self.stem = Stem(
            in_chs=in_chans,
            out_chs=embed_dim,
            padding=cfg.conv_cfg.padding,
            bias=cfg.stem_bias,
            act_layer=cfg.conv_cfg.act_layer,
            norm_layer=cfg.conv_cfg.norm_layer,
            norm_eps=cfg.conv_cfg.norm_eps,
            fs_time=fs_time,
        )
        stride = 1
        self.feature_info += [dict(num_chs=self.stem.out_chs,
                                    reduction=2, module='stem')]
        feat_size = tuple(
            [i // s for i, s in zip(img_size, to_2tuple(stride))])

        num_stages = len(cfg.embed_dim)
        assert len(cfg.depths) == num_stages
        dpr = [x.tolist() for x in torch.linspace(
            0, drop_path_rate, sum(cfg.depths)).split(cfg.depths)]
        in_chs = self.stem.out_chs
        stages = []

        if print_cfg: print(cfg.conv_cfg)
        for i in range(num_stages):
            if i in [0]:
                stage_stride = 2
            else:
                stage_stride = 2
            if i == 0:
                transformer_cfg.window_size = (103, 24)
                transformer_cfg.grid_size = (103, 24)
            elif i == 1:
                transformer_cfg.window_size = (103, 12)
                transformer_cfg.grid_size =(103, 12)
            elif i == 2:
                transformer_cfg.window_size = (103, 3)
                transformer_cfg.grid_size = (103, 3)


            out_chs = cfg.embed_dim[i]
            feat_size = tuple([(r - 1) // stage_stride + 1 for r in feat_size])
            if print_cfg: print('feat_size', feat_size)
            stages += [MaxxVitStage(
                in_chs,
                out_chs,
                depth=cfg.depths[i],
                block_types=cfg.block_type[i],
                conv_cfg=cfg.conv_cfg,
                transformer_cfg=transformer_cfg,
                feat_size=feat_size,
                drop_path=dpr[i],
            )]
            stride *= stage_stride
            if print_cfg: print('stride', stride)
            in_chs = out_chs
            self.feature_info += [dict(num_chs=out_chs,
                                       reduction=stride, module=f'stages.{i}')]
        self.stages = nn.Sequential(*stages)


        ### regression specific convolution and attention block
        if print_cfg:
            print('stages regression feature size', feat_size)
            print('stages regression cfg.conv_cfg', cfg.conv_cfg)
        cfg.conv_cfg.kernel_size = 3
        transformer_cfg.window_size = (155, 3)
        transformer_cfg.grid_size = (1, 1)
        self.num_features = 2048
        stages_regression = [ConstantPad2d((0, 0, 0, 1), 0),
                             MaxxVitStage(
            1024,
            self.num_features,
            depth=1,
            block_types='M',
            conv_cfg=cfg.conv_cfg,
            transformer_cfg=transformer_cfg,
            feat_size=feat_size,
            drop_path=[0]*2,
        )]

        stride *= stage_stride
        if print_cfg: print('stride', stride)
        in_chs = out_chs
        self.feature_info += [dict(num_chs=out_chs,
                                    reduction=stride, module=f'stages_regression.{i}')]

        self.regression_vit = nn.Sequential(*stages_regression)

        final_norm_layer = partial(get_norm_layer(
            cfg.transformer_cfg.norm_layer), eps=cfg.transformer_cfg.norm_eps)
        self.head_hidden_size = cfg.head_hidden_size

        self.norm = nn.Identity()
        
        self.final_latent_space_mlp = NormMlpClassifierHead2(
            self.num_features,
            out_features=self.dim_final_latent_space,
            pool_type=global_pool,
            drop_rate=drop_rate,
            norm_layer=final_norm_layer,
            n_covariates=n_covariates
        )
            
        self.heads = nn.ModuleDict({
            f"final_head_{name}": nn.Linear(self.dim_final_latent_space, output_dim)
            for name, output_dim in zip(target_names, target_output_dims)
        })
    
        # decoder:
        n_chs = out_chs
        transformer_cfg.window_size = (103, 3)
        transformer_cfg.grid_size = (103, 3)
        self.decoder_sleepstages = MaxVit_SleepStagesDecoder(n_chs, cfg, transformer_cfg, feat_size, fs_time=fs_time)

        # Weight init (default PyTorch init works well for AdamW if scheme not set)
        assert cfg.weight_init in (
            '', 'normal', 'trunc_normal', 'xavier_normal', 'vit_eff')
        if cfg.weight_init:
            named_apply(partial(self._init_weights,
                        scheme=cfg.weight_init), self)

    def _init_weights(self, module, name, scheme=''):
        if hasattr(module, 'init_weights'):
            try:
                module.init_weights(scheme=scheme)
            except TypeError:
                module.init_weights()

    @torch.jit.ignore
    def no_weight_decay(self):
        return {
            k for k, _ in self.named_parameters()
            if any(n in k for n in ["relative_position_bias_table", "rel_pos.mlp"])}

    @torch.jit.ignore
    def group_matcher(self, coarse=False):
        matcher = dict(
            stem=r'^stem',  # stem and embed
            blocks=[(r'^stages\.(\d+)', None), (r'^norm', (99999,))]
        )
        return matcher

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        for s in self.stages:
            s.grad_checkpointing = enable

    @torch.jit.ignore
    def get_classifier(self):
        return self.final_latent_space_mlp.fc

    def forward_maxvit(self, x):
        x = self.stages(x)
        x = self.norm(x)
        return x

    def forward_regression(self, x, covariates, pre_logits: bool = False):
        x = self.final_layer(x)
        return x
    
    def forward_final_latent_space(self, x, covariates):
        """ 
        Forward pass for the final latent space (before the final linear layers for regression and classification).
        """
        x = self.regression_vit(x)
        x = self.final_latent_space_mlp(x, covariates)
        return x
    
    def forward_sleepstages(self, x, skip_connections):
        return self.decoder_sleepstages(x, skip_connections)
        

    def forward(self, x, covariates, return_features_lhl=False):
            
        x1, x2, x3, x = self.stem(x)
        skip_connections = [x1, x2, x3]
        x = self.forward_maxvit(x)

        yp_stages = self.forward_sleepstages(x, skip_connections)
        
        # Map to final latent space and then do the final regression and classification heads.
        features_lhl = self.forward_final_latent_space(x, covariates)
        
        # Now, run the final regression and classification heads, assemble all the different outputs, 
        # collect them in a dictionary and return them.
        
        outputs = {name.replace('final_head_', ''): head(features_lhl) for name, head in self.heads.items()}
                
        yp_regression = []
        yp_classification = []
        for name, output_dim, task_type in zip(self.target_names, self.target_output_dims, self.task_types):
            if task_type == 'regression':
                yp_regression.append(outputs[name])
            elif task_type == 'classification':
                yp_classification.append(outputs[name])
            else:
                raise ValueError(f'Unknown task type: {task_type}')   
        
        yp_regression = torch.cat(yp_regression, dim=1)
        yp_classification = torch.cat(yp_classification, dim=1)
        
        if return_features_lhl:
            return yp_regression, yp_classification, yp_stages, features_lhl

        return yp_regression, yp_classification, yp_stages


    def set_trainable_heads(self, names, is_trainable=True):
        """Set specific heads as trainable or not based on their names."""
        for name in names:
            for param in self.heads[name].parameters():
                param.requires_grad = is_trainable
                
