# Copyright (c) OpenMMLab. All rights reserved.
import warnings

import torch
import torch.nn as nn
from mmdet.models.losses import smooth_l1_loss
from mmdet.models.losses.utils import reduce_loss
from ..builder import ROTATED_LOSSES


def probabilistic_l1_loss(pred,
                          target,
                          bbox_cov,
                          weight=None,
                          beta=1.0,
                          reduction=None,
                          avg_factor=None,
                          **kwargs
                          ):
    """Smooth L1 loss.

    Args:
        weight:
        avg_factor:
        reduction:
        bbox_cov:
        pred (torch.Tensor): The prediction.
        target (torch.Tensor): The learning target of the prediction.
        beta (float, optional): The threshold in the piecewise function.
            Defaults to 1.0.

    Returns:
        torch.Tensor: Calculated loss
    """
    bbox_cov = torch.clamp(bbox_cov, -7.0, 7.0)
    loss_box_reg = 0.5 * torch.exp(-bbox_cov) * _smooth_l1_loss(
        pred,
        target,
        beta=beta,
    )
    loss_covariance_regularize = 0.5 * bbox_cov
    loss_box_reg += loss_covariance_regularize

    if weight is not None:
        loss_box_reg = loss_box_reg * weight

    # if avg_factor is not specified, just reduce the loss
    if avg_factor is None:
        loss_box_reg = reduce_loss(loss_box_reg, reduction)
    else:
        # if reduction is mean, then average the loss by avg_factor
        if reduction == 'mean':
            # Avoid causing ZeroDivisionError when avg_factor is 0.0,
            # i.e., all labels of an image belong to ignore index.
            eps = torch.finfo(torch.float32).eps
            loss_box_reg = loss_box_reg.sum() / (avg_factor + eps)
        # if reduction is 'none', then do nothing, otherwise raise an error
        elif reduction != 'none':
            raise ValueError('avg_factor can not be used with reduction="sum"')
    return loss_box_reg


def _smooth_l1_loss(pred, target, beta=1.0):
    """Smooth L1 loss.

    Args:
        pred (torch.Tensor): The prediction.
        target (torch.Tensor): The learning target of the prediction.
        beta (float, optional): The threshold in the piecewise function.
            Defaults to 1.0.

    Returns:
        torch.Tensor: Calculated loss
    """
    assert beta > 0
    if target.numel() == 0:
        return pred.sum() * 0

    assert pred.size() == target.size()
    diff = torch.abs(pred - target)
    loss = torch.where(diff < beta, 0.5 * diff * diff / beta,
                       diff - 0.5 * beta)
    return loss


@ROTATED_LOSSES.register_module()
class ProbabilisticL1Loss(nn.Module):
    """RotatedIoULoss.

    Computing the IoU loss between a set of predicted rbboxes and
    target rbboxes.
    Args:
        linear (bool): If True, use linear scale of loss else determined
            by mode. Default: False.
        eps (float): Eps to avoid log(0).
        reduction (str): Options are "none", "mean" and "sum".
        loss_weight (float): Weight of loss.
        mode (str): Loss scaling mode, including "linear", "square", and "log".
            Default: 'log'
    """

    def __init__(self, beta=1.0, reduction='mean', loss_weight=1.0):
        super(ProbabilisticL1Loss, self).__init__()
        self.beta = beta
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self,
                pred,
                target,
                bbox_conv,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                compute_bbox_conv=True,
                **kwargs):
        """Smooth L1 loss.

        Args:
            compute_bbox_conv:
            reduction_override:
            avg_factor:
            bbox_conv:
            weight:
            target:
            pred:
        """
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        if compute_bbox_conv:
            loss_bbox = self.loss_weight * probabilistic_l1_loss(
                pred,
                target,
                bbox_conv,
                weight,
                beta=self.beta,
                reduction=reduction,
                avg_factor=avg_factor,
                **kwargs)
        else:
            loss_bbox = self.loss_weight * smooth_l1_loss(
                pred,
                target,
                weight,
                beta=self.beta,
                reduction=reduction,
                avg_factor=avg_factor,
                **kwargs)
        return loss_bbox
