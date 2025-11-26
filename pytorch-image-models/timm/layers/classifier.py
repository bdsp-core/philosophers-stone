""" Classifier head and layer factory

Hacked together by / Copyright 2020 Ross Wightman
"""
from collections import OrderedDict
from functools import partial
from typing import Optional, Union, Callable

import torch
import torch.nn as nn
from torch.nn import functional as F

from .adaptive_avgmax_pool import SelectAdaptivePool2d
from .create_act import get_act_layer
from .create_norm import get_norm_layer


def _create_pool(
        num_features: int,
        num_classes: int,
        pool_type: str = 'avg',
        use_conv: bool = False,
        input_fmt: Optional[str] = None,
):
    flatten_in_pool = not use_conv  # flatten when we use a Linear layer after pooling
    if not pool_type:
        assert num_classes == 0 or use_conv,\
            'Pooling can only be disabled if classifier is also removed or conv classifier is used'
        flatten_in_pool = False  # disable flattening if pooling is pass-through (no pooling)
    global_pool = SelectAdaptivePool2d(
        pool_type=pool_type,
        flatten=flatten_in_pool,
        input_fmt=input_fmt,
    )
    num_pooled_features = num_features * global_pool.feat_mult()
    return global_pool, num_pooled_features


def _create_fc(num_features, num_classes, use_conv=False):
    if num_classes <= 0:
        fc = nn.Identity()  # pass-through (no classifier)
    elif use_conv:
        fc = nn.Conv2d(num_features, num_classes, 1, bias=True)
    else:
        fc = nn.Linear(num_features, num_classes, bias=True)
    return fc


def create_classifier(
        num_features: int,
        num_classes: int,
        pool_type: str = 'avg',
        use_conv: bool = False,
        input_fmt: str = 'NCHW',
):
    global_pool, num_pooled_features = _create_pool(
        num_features,
        num_classes,
        pool_type,
        use_conv=use_conv,
        input_fmt=input_fmt,
    )
    fc = _create_fc(
        num_pooled_features,
        num_classes,
        use_conv=use_conv,
    )
    return global_pool, fc


class ClassifierHead(nn.Module):
    """Classifier head w/ configurable global pooling and dropout."""

    def __init__(
            self,
            in_features: int,
            num_classes: int,
            pool_type: str = 'avg',
            drop_rate: float = 0.,
            use_conv: bool = False,
            input_fmt: str = 'NCHW',
    ):
        """
        Args:
            in_features: The number of input features.
            num_classes:  The number of classes for the final classifier layer (output).
            pool_type: Global pooling type, pooling disabled if empty string ('').
            drop_rate: Pre-classifier dropout rate.
        """
        super(ClassifierHead, self).__init__()
        self.in_features = in_features
        self.use_conv = use_conv
        self.input_fmt = input_fmt

        global_pool, fc = create_classifier(
            in_features,
            num_classes,
            pool_type,
            use_conv=use_conv,
            input_fmt=input_fmt,
        )
        self.global_pool = global_pool
        self.drop = nn.Dropout(drop_rate)
        self.fc = fc
        self.flatten = nn.Flatten(1) if use_conv and pool_type else nn.Identity()

    def reset(self, num_classes, pool_type=None):
        if pool_type is not None and pool_type != self.global_pool.pool_type:
            self.global_pool, self.fc = create_classifier(
                self.in_features,
                num_classes,
                pool_type=pool_type,
                use_conv=self.use_conv,
                input_fmt=self.input_fmt,
            )
            self.flatten = nn.Flatten(1) if self.use_conv and pool_type else nn.Identity()
        else:
            num_pooled_features = self.in_features * self.global_pool.feat_mult()
            self.fc = _create_fc(
                num_pooled_features,
                num_classes,
                use_conv=self.use_conv,
            )

    def forward(self, x, pre_logits: bool = False):
        x = self.global_pool(x)
        x = self.drop(x)
        if pre_logits:
            return self.flatten(x)
        x = self.fc(x)
        return self.flatten(x)


class NormMlpClassifierHead(nn.Module):

    def __init__(
            self,
            in_features: int,
            num_classes: int,
            hidden_size: Optional[int] = None,
            pool_type: str = 'avg',
            drop_rate: float = 0.,
            norm_layer: Union[str, Callable] = 'layernorm2d',
            act_layer: Union[str, Callable] = 'tanh',
            n_covariates: int = 0,
    ):
        """
        Args:
            in_features: The number of input features.
            num_classes:  The number of classes for the final classifier layer (output).
            hidden_size: The hidden size of the MLP (pre-logits FC layer) if not None.
            pool_type: Global pooling type, pooling disabled if empty string ('').
            drop_rate: Pre-classifier dropout rate.
            norm_layer: Normalization layer type.
            act_layer: MLP activation layer type (only used if hidden_size is not None).
        """
        super().__init__()
        self.in_features = in_features
        self.hidden_size = hidden_size
        self.num_features = in_features
        self.use_conv = not pool_type
        norm_layer = get_norm_layer(norm_layer)
        act_layer = get_act_layer(act_layer)
        linear_layer = partial(nn.Conv2d, kernel_size=1) if self.use_conv else nn.Linear

        self.global_pool = SelectAdaptivePool2d(pool_type=pool_type)
        self.norm = norm_layer(in_features)
        self.flatten = nn.Flatten(1) if pool_type else nn.Identity()
        if hidden_size:
            self.pre_logits = nn.Sequential(OrderedDict([
                ('fc', linear_layer(in_features + n_covariates, hidden_size)),
                ('act', act_layer()),
            ]))
            self.num_features = hidden_size
        else:
            self.pre_logits = nn.Identity()
        self.drop = nn.Dropout(drop_rate)
        self.fc = linear_layer(self.num_features, num_classes) if num_classes > 0 else nn.Identity()

    def reset(self, num_classes, global_pool=None):
        if global_pool is not None:
            self.global_pool = SelectAdaptivePool2d(pool_type=global_pool)
            self.flatten = nn.Flatten(1) if global_pool else nn.Identity()
        self.use_conv = self.global_pool.is_identity()
        linear_layer = partial(nn.Conv2d, kernel_size=1) if self.use_conv else nn.Linear
        if self.hidden_size:
            if ((isinstance(self.pre_logits.fc, nn.Conv2d) and not self.use_conv) or
                    (isinstance(self.pre_logits.fc, nn.Linear) and self.use_conv)):
                with torch.no_grad():
                    new_fc = linear_layer(self.in_features, self.hidden_size)
                    new_fc.weight.copy_(self.pre_logits.fc.weight.reshape(new_fc.weight.shape))
                    new_fc.bias.copy_(self.pre_logits.fc.bias)
                    self.pre_logits.fc = new_fc
        self.fc = linear_layer(self.num_features, num_classes) if num_classes > 0 else nn.Identity()

    def forward(self, x, covariates, pre_logits: bool = False):
        # print('regression head forward', x.shape, covariates.shape)
        x = self.global_pool(x)
        x = self.norm(x)
        x = self.flatten(x)
        # add covariates: e.g. sex or one-hot encoding for cohort added to fully connected layer
        x = torch.cat((x, covariates), dim=1)
        x = self.pre_logits(x)
        x = self.drop(x)
        if pre_logits:
            return x
        x = self.fc(x)
        return x
    



class NormMlpClassifierHead2(nn.Module):
    """ Just like NormMlpClassifierHead but with a different forward method without a final linear layer. """

    def __init__(
            self,
            in_features: int,
            out_features: int,
            pool_type: str = 'avg',
            drop_rate: float = 0.,
            norm_layer: Union[str, Callable] = 'layernorm2d',
            act_layer: Union[str, Callable] = 'tanh',
            n_covariates: int = 0,
    ):
        """
        Args:
            in_features: The number of input features.
            num_classes:  The number of classes for the final classifier layer (output).
            hidden_size: The hidden size of the MLP (pre-logits FC layer) if not None.
            pool_type: Global pooling type, pooling disabled if empty string ('').
            drop_rate: Pre-classifier dropout rate.
            norm_layer: Normalization layer type.
            act_layer: MLP activation layer type (only used if hidden_size is not None).
        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_features = in_features
        self.use_conv = not pool_type
        norm_layer = get_norm_layer(norm_layer)
        act_layer = get_act_layer(act_layer)
        linear_layer = partial(nn.Conv2d, kernel_size=1) if self.use_conv else nn.Linear

        self.global_pool = SelectAdaptivePool2d(pool_type=pool_type)
        self.norm = norm_layer(in_features)
        self.flatten = nn.Flatten(1) if pool_type else nn.Identity()
        if out_features:
            self.pre_logits = nn.Sequential(OrderedDict([
                ('fc', linear_layer(in_features + n_covariates, out_features)),
                ('act', act_layer()),
            ]))
            self.num_features = out_features
        else:
            self.pre_logits = nn.Identity()
        self.drop = nn.Dropout(drop_rate)

    def reset(self, num_classes, global_pool=None):
        if global_pool is not None:
            self.global_pool = SelectAdaptivePool2d(pool_type=global_pool)
            self.flatten = nn.Flatten(1) if global_pool else nn.Identity()
        self.use_conv = self.global_pool.is_identity()
        linear_layer = partial(nn.Conv2d, kernel_size=1) if self.use_conv else nn.Linear
        if self.out_features:
            if ((isinstance(self.pre_logits.fc, nn.Conv2d) and not self.use_conv) or
                    (isinstance(self.pre_logits.fc, nn.Linear) and self.use_conv)):
                with torch.no_grad():
                    new_fc = linear_layer(self.in_features, self.out_features)
                    new_fc.weight.copy_(self.pre_logits.fc.weight.reshape(new_fc.weight.shape))
                    new_fc.bias.copy_(self.pre_logits.fc.bias)
                    self.pre_logits.fc = new_fc

    def forward(self, x, covariates):
        # print('regression head forward', x.shape, covariates.shape)
        x = self.global_pool(x)
        x = self.norm(x)
        x = self.flatten(x)
        # add covariates: e.g. sex or one-hot encoding for cohort added to fully connected layer
        # print("MLP input shape", x.shape, covariates.shape, covariates)
        x = torch.cat((x, covariates), dim=1)
        # print("MLP post-cat shape", x.shape, x[-2:])
        x = self.pre_logits(x)
        # print("MLP output shape", x.shape, x[-2:])
        x = self.drop(x)

        return x

