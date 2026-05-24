import torch
import random
import numpy as np
from pytorch_lightning import LightningModule
from torch.nn import Module, Linear, ReLU, Sequential, Conv2d, MaxPool2d, BatchNorm2d, Upsample, ConvTranspose2d, ConstantPad2d
import torch.optim as optim
from loss_fct import calculate_loss, calculate_loss_MLT
from torch.optim.lr_scheduler import LambdaLR


torch.manual_seed(42)
random.seed(69)
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


class BaseOracleModule(LightningModule):
    """
    Base class for oracle models. Implements the training, validation and test steps,
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


class Sleep_Oracle_CNN_based(BaseOracleModule):
    def __init__(self,
                 lr, cost_function, n_targets, target_weights_regression, cf_stage_weight, ntr,
                 cnn_layers, linear_layers, sleepstages_layer_decoder):
        super().__init__(lr, cost_function, n_targets,
                         target_weights_regression, cf_stage_weight, ntr)
        # ignore=['cnn_layers', 'linear_layers', 'sleepstages_layer_decoder'])
        self.save_hyperparameters()

        self.cnn_layers = cnn_layers
        self.linear_layers = linear_layers
        self.sleepstages_layer_decoder = sleepstages_layer_decoder

    # Defining the forward pass

    def forward(self, x):
        verbose = False
        if verbose:
            print(x.shape)
        x = self.cnn_layers(x)
        if verbose:
            print(x.shape)
        yp_regression = self.linear_layers(x.view(x.size(0), -1))
        if verbose:
            print(yp_regression.shape)

        # Decoder either needs to return 39600 or 39600//30.
        # We predict on 30-second resolution, so if it returns 39600, we reduce it to 39600/30 here as well.
        yp_stages = self.sleepstages_layer_decoder(x)
        annot_len = 39600
        # if yp_stages.shape[-1] == 39600:
        yp_stages = yp_stages[:, :, :, :annot_len//30]
        # elif yp_stages.shape[-1] == 39600//30:
        # pass
        # else:
        # raise ValueError(f"yp_stages.shape ={yp_stages.shape}. Last dimension != 39600 and != 39600 //30 after decoder")

        if verbose:
            print(yp_stages.shape)
        # BxCx1xannot_len --> BxCxannot_len
        yp_stages = yp_stages.squeeze(dim=2)

        if verbose:
            print(yp_stages.shape)

        return yp_regression, yp_stages

def create_original_cnn_layers():
    # Construct the cnn_layers for the original model and return them
    return Sequential(
        # Defining a 2D convolution layer
        Conv2d(1, 3, kernel_size=(7, 1), stride=1, padding=1),
        BatchNorm2d(3),
        ReLU(inplace=True),
        Conv2d(3, 3, kernel_size=(7, 1), stride=1, padding=1),
        BatchNorm2d(3),
        ReLU(inplace=True),
        MaxPool2d(kernel_size=(7, 1), stride=(7, 1)),

        Conv2d(3, 9, kernel_size=(7, 1), stride=1, padding=1),
        BatchNorm2d(9),
        ReLU(inplace=True),
        Conv2d(9, 9, kernel_size=(7, 1), stride=1, padding=1),
        BatchNorm2d(9),
        ReLU(inplace=True),
        MaxPool2d(kernel_size=(7, 1), stride=(7, 1)),

        Conv2d(9, 27, kernel_size=(5, 1), stride=1, padding=1),
        BatchNorm2d(27),
        ReLU(inplace=True),
        Conv2d(27, 27, kernel_size=(5, 1), stride=1, padding=1),
        BatchNorm2d(27),
        ReLU(inplace=True),
        MaxPool2d(kernel_size=(5, 1), stride=(5, 1)),

        Conv2d(27, 81, kernel_size=(3, 2), stride=1, padding=1),
        BatchNorm2d(81),
        ReLU(inplace=True),
        Conv2d(81, 81, kernel_size=(3, 2), stride=1, padding=1),
        BatchNorm2d(81),
        ReLU(inplace=True),
        MaxPool2d(kernel_size=(3, 1), stride=(3, 1)),

        Conv2d(81, 162, kernel_size=(3, 3), stride=1, padding=1),
        BatchNorm2d(162),
        ReLU(inplace=True),
        Conv2d(162, 162, kernel_size=(3, 3), stride=1, padding=1),
        BatchNorm2d(162),
        ReLU(inplace=True),
        MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),

        Conv2d(162, 324, kernel_size=3, stride=1, padding=1),
        BatchNorm2d(324),
        ReLU(inplace=True),
        Conv2d(324, 324, kernel_size=3, stride=1, padding=1),
        BatchNorm2d(324),
        ReLU(inplace=True),
        MaxPool2d(kernel_size=(3, 2), stride=(3, 2)),

        Conv2d(324, 648, kernel_size=3, stride=1, padding=1),
        BatchNorm2d(648),
        ReLU(inplace=True),
        Conv2d(648, 648, kernel_size=3, stride=1, padding=1),
        BatchNorm2d(648),
        ReLU(inplace=True),
        MaxPool2d(kernel_size=(3, 3), stride=(3, 3)),
    )

def create_original_linear_layers(n_targets_regression):
    # Construct the linear_layers for the original model and return them

    n_units = 11664
    return Sequential(
        Linear(n_units, n_targets)
    )

def create_original_sleepstages_layer_decoder():
    # Construct the sleepstages_layer_decoder for the original model and return them
    return Sequential(
        Upsample(scale_factor=(3, 3), mode='bilinear', align_corners=True),
        Conv2d(648, 324, kernel_size=3, padding=1, bias=False),
        BatchNorm2d(324),
        ReLU(inplace=True),
        Conv2d(324, 324, kernel_size=3, padding=1, bias=False),
        BatchNorm2d(324),
        ReLU(inplace=True),

        Upsample(scale_factor=(1, 3), mode='bilinear', align_corners=True),
        Conv2d(324, 162, kernel_size=3, padding=1, bias=False),
        BatchNorm2d(162),
        ReLU(inplace=True),
        Conv2d(162, 162, kernel_size=3, padding=1, bias=False),
        BatchNorm2d(162),
        ReLU(inplace=True),

        Upsample(scale_factor=(1, 2), mode='bilinear', align_corners=True),
        Conv2d(162, 81, kernel_size=3, padding=1, bias=False),
        BatchNorm2d(81),
        ReLU(inplace=True),
        Conv2d(81, 81, kernel_size=3, padding=1, bias=False),
        BatchNorm2d(81),
        ReLU(inplace=True),

        Upsample(scale_factor=(1, 2), mode='bilinear', align_corners=True),
        Conv2d(81, 27, kernel_size=3, padding=1, bias=False),
        BatchNorm2d(27),
        ReLU(inplace=True),
        Conv2d(27, 27, kernel_size=3, padding=1, bias=False),
        BatchNorm2d(27),
        ReLU(inplace=True),

        Upsample(scale_factor=(1, 2), mode='bilinear', align_corners=True),
        Conv2d(27, 9, kernel_size=3, padding=1, bias=False),
        BatchNorm2d(9),
        ReLU(inplace=True),
        Conv2d(9, 9, kernel_size=3, padding=1, bias=False),
        BatchNorm2d(9),
        ReLU(inplace=True),

        Upsample(scale_factor=(1, 2), mode='bilinear', align_corners=True),
        Conv2d(9, 3, kernel_size=3, padding=1, bias=False),
        BatchNorm2d(3),
        ReLU(inplace=True),
        Conv2d(3, 3, kernel_size=3, padding=1, bias=False),
        BatchNorm2d(3),
        ReLU(inplace=True),

        Upsample(scale_factor=(1, 2), mode='bilinear', align_corners=True),
        Conv2d(3, 1, kernel_size=3, padding=1, bias=False),
        Reshape(6, 1, -1),  # Bx1x6xN --> Bx6x1xN
    )

def create_simple_cnn_layers():
    # Construct the cnn_layers for the SimpleCNN model and return them
    return Sequential(
        Conv2d(1, 16, kernel_size=(5, 2), stride=(5, 1)),
        ReLU(inplace=True),
        Conv2d(16, 32, kernel_size=(5, 2), stride=(5, 1)),
        ReLU(inplace=True),
        Conv2d(32, 64, kernel_size=(5, 2), stride=(5, 2)),
        ReLU(inplace=True),
        Conv2d(64, 128, kernel_size=(5, 2), stride=(5, 2)),
        ReLU(inplace=True),
        Conv2d(128, 256, kernel_size=(3, 3), stride=(3, 3)),
        ReLU(inplace=True),
        Conv2d(256, 512, kernel_size=(3, 3), stride=(2, 2)),
        ReLU(inplace=True),
        Conv2d(512, 1024, kernel_size=(3, 3), stride=(2, 2)),
        ReLU(inplace=True),
        Conv2d(1024, 2048, kernel_size=(3, 1), stride=(2, 1)),
        Reshape(1, 1, -1),  # BxCxHxW -> Bx1x1xC*H*W
        # torch size after last Conv2d is (-1, 2048, 1, 1)
        # reshape makes it (-1, 1, 1, 2048)
    )

def create_simple_linear_layers(n_targets_regression):
    # Construct the linear_layers for the SimpleCNN model and return them
    return Sequential(
        Linear(2048, 256),
        ReLU(inplace=True),
        Linear(256, 128),
        ReLU(inplace=True),
        Linear(128, n_targets),
    )

def create_simple_sleepstages_layer_decoder():
    # Simple sleepstage layer decoder
    return Sequential(
        Linear(2048, 39600 // 30),
        # from (-1, 1, 1, 2048) to (-1, 6, 1, 2048):
        Conv2d(1, 6, kernel_size=1, stride=1, padding=0)
    )

class SimpleCNNEncoder(Module):
    def __init__(self):
        super(SimpleCNNEncoder, self).__init__()
        self.layer1 = Sequential(
            Conv2d(1, 16, kernel_size=(5, 2), stride=(5, 1)),
            ReLU(inplace=True),
        )
        self.layer2 = Sequential(
            Conv2d(16, 32, kernel_size=(5, 2), stride=(5, 1)),
            ReLU(inplace=True),
        )
        self.layer3 = Sequential(
            Conv2d(32, 64, kernel_size=(5, 2), stride=(5, 2)),
            ReLU(inplace=True),
        )
        self.layer4 = Sequential(
            Conv2d(64, 128, kernel_size=(5, 2), stride=(5, 2)),
            ReLU(inplace=True),
        )
        self.layer5 = Sequential(
            Conv2d(128, 256, kernel_size=(3, 3), stride=(3, 3)),
            ReLU(inplace=True),
        )
        self.layer6 = Sequential(
            Conv2d(256, 512, kernel_size=(3, 3), stride=(2, 2)),
            ReLU(inplace=True),
        )
        self.layer7 = Sequential(
            Conv2d(512, 1024, kernel_size=(3, 3), stride=(2, 2)),
            ReLU(inplace=True),
        )
        self.layer8 = Sequential(
            Conv2d(1024, 2048, kernel_size=(3, 1), stride=(2, 1)),
        )

    def forward(self, x):
        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        x5 = self.layer5(x4)
        x6 = self.layer6(x5)
        x7 = self.layer7(x6)
        x8 = self.layer8(x7)

        return x1, x2, x3, x4, x5, x6, x7, x8

class SimpleCNNDecoder(Module):
    def __init__(self):
        super(SimpleCNNDecoder, self).__init__()
        self.layer1 = Sequential(
            # Add decoder layers here (usually ConvTranspose2d layers)
            ConvTranspose2d(2048, 1024, kernel_size=(3, 1), stride=(2, 1)),
            ReLU(inplace=True),
            ConstantPad2d((0, 0, 1, 0), 0),
        )
        self.layer2 = Sequential(
            ConvTranspose2d(1024 * 2, 512, kernel_size=(3, 3), stride=(2, 2)),
            ReLU(inplace=True),
            ConstantPad2d((0, 0, 1, 0), 0),
        )
        self.layer3 = Sequential(
            ConvTranspose2d(512 * 2, 256, kernel_size=(3, 3), stride=(2, 2)),
            ReLU(inplace=True),
            ConstantPad2d((0, 1, 0, 0), 0),
        )
        self.layer4 = Sequential(
            ConvTranspose2d(256 * 2, 128, kernel_size=(3, 3), stride=(3, 3)),
            ReLU(inplace=True),
            ConstantPad2d((0, 0, 0, 0), 0),
        )
        self.layer5 = Sequential(
            ConvTranspose2d(128 * 2, 64, kernel_size=(5, 2), stride=(5, 2)),
            ReLU(inplace=True),
            ConstantPad2d((0, 1, 1, 0), 0),
        )
        self.layer6 = Sequential(
            ConvTranspose2d(64 * 2, 32, kernel_size=(5, 2), stride=(5, 2)),
            ReLU(inplace=True),
            ConstantPad2d((0, 0, 4, 0), 0),
        )
        self.layer7 = Sequential(
            ConvTranspose2d(32 * 2, 16, kernel_size=(5, 2), stride=(5, 1)),
            ReLU(inplace=True),
            ConstantPad2d((0, 0, 0, 0), 0),
        )
        self.layer8 = Sequential(
            ConvTranspose2d(16 * 2, 1, kernel_size=(5, 2), stride=(5, 1)),
            ReLU(inplace=True),
            ConstantPad2d((0, 0, 0, 0), 0),
        )
        # one scalaer per 30 seconds data, use Conv2d with (30, 100) kernel and stride 30
        self.layer_final = Sequential(
            Conv2d(1, 6, kernel_size=(30, 100), stride=30),
        )

    def forward(self, x, skip_outputs):
        verbose = False
        x = self.layer1(x)
        # Combine with the corresponding skip connection
        x = torch.cat((x, skip_outputs[-1]), 1)
        x = self.layer2(x)
        # Combine with the corresponding skip connection
        x = torch.cat((x, skip_outputs[-2]), 1)
        x = self.layer3(x)
        # Combine with the corresponding skip connection
        x = torch.cat((x, skip_outputs[-3]), 1)
        x = self.layer4(x)
        # Combine with the corresponding skip connection
        x = torch.cat((x, skip_outputs[-4]), 1)
        x = self.layer5(x)
        # Combine with the corresponding skip connection
        x = torch.cat((x, skip_outputs[-5]), 1)
        x = self.layer6(x)
        # Combine with the corresponding skip connection
        x = torch.cat((x, skip_outputs[-6]), 1)
        x = self.layer7(x)
        # Combine with the corresponding skip connection
        x = torch.cat((x, skip_outputs[-7]), 1)
        x = self.layer8(x)
        x = self.layer_final(x)

        return x

class SimpleUNet(BaseOracleModule):
    def __init__(self, lr, cost_function, n_targets, target_weights_regression, cf_stage_weight, ntr):
        super().__init__(lr, cost_function, n_targets,
                         target_weights_regression, cf_stage_weight, ntr)

        self.encoder = SimpleCNNEncoder()
        self.linear_layers = create_simple_linear_layers(n_targets)
        self.sleepstages_layer_decoder = SimpleCNNDecoder()

    def forward(self, x):
        verbose = False
        encoder_output = self.encoder(x)
        x = encoder_output[-1]
        skip_outputs = encoder_output[:-1]

        yp_stages = self.sleepstages_layer_decoder(x, skip_outputs)
        # Flatten the feature maps, BxCxHxW -> BxCx(HxW)
        yp_stages = yp_stages.view(yp_stages.size(0), yp_stages.size(1),  -1)
        x = x.view(x.size(0), -1)
        yp_regression = self.linear_layers(x)


        return yp_regression, yp_stages
