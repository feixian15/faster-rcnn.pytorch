from __future__ import absolute_import
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from model.utils.config import cfg
from .proposal_layer import _ProposalLayer
from .anchor_target_layer import _AnchorTargetLayer
from model.utils.net_utils import _smooth_l1_loss

import numpy as np
import math
import pdb
import time

class _RPN(nn.Module):
    """ region proposal network """
    def __init__(self, din):
        super(_RPN, self).__init__()
        
        self.din = din  # get depth of input feature map, e.g., 512
        self.anchor_scales = cfg.ANCHOR_SCALES
        self.anchor_ratios = cfg.ANCHOR_RATIOS
        self.feat_stride = cfg.FEAT_STRIDE[0]

        # define the convrelu layers processing input feature map
        self.RPN_Conv = nn.Conv2d(self.din, 512, 3, 1, 1, bias=True)

        # define bg/fg classifcation score layer
        self.nc_score_out = len(self.anchor_scales) * len(self.anchor_ratios) * 2 # 2(bg/fg) * 9 (anchors)
        self.RPN_cls_score = nn.Conv2d(512, self.nc_score_out, 1, 1, 0)

        # define anchor box offset prediction layer
        self.nc_bbox_out = len(self.anchor_scales) * len(self.anchor_ratios) * 4 # 4(coords) * 9 (anchors)
        self.RPN_bbox_pred = nn.Conv2d(512, self.nc_bbox_out, 1, 1, 0)

        # define proposal layer
        self.RPN_proposal = _ProposalLayer(self.feat_stride, self.anchor_scales, self.anchor_ratios)

        # define anchor target layer
        self.RPN_anchor_target = _AnchorTargetLayer(self.feat_stride, self.anchor_scales, self.anchor_ratios)

        self.rpn_loss_cls = 0
        self.rpn_loss_box = 0

    @staticmethod
    def reshape(x, d):
        input_shape = x.size()
        x = x.view(
            input_shape[0],
            int(d),
            int(float(input_shape[1] * input_shape[2]) / float(d)),
            input_shape[3]
        )
        return x

    def forward(self, base_feat, im_info, gt_boxes, num_boxes):

        batch_size = base_feat.size(0)

        # return feature map after convrelu layer
        rpn_conv1 = F.relu(self.RPN_Conv(base_feat), inplace=True)
        # get rpn classification score
        rpn_cls_score = self.RPN_cls_score(rpn_conv1)
        rpn_cls_score_reshape = self.reshape(rpn_cls_score, 2)
        rpn_cls_prob_reshape = F.softmax(rpn_cls_score_reshape, 1)
        rpn_cls_prob = self.reshape(rpn_cls_prob_reshape, self.nc_score_out)

        # get rpn offsets to the anchor boxes
        rpn_bbox_pred = self.RPN_bbox_pred(rpn_conv1)

        # proposal layer
        cfg_key = 'TRAIN' if self.training else 'TEST'
        
        rois = self.RPN_proposal((rpn_cls_prob.data, rpn_bbox_pred.data,
                                 im_info, cfg_key))
        
        self.rpn_loss_cls = 0
        self.rpn_loss_box = 0

        # generating training labels and build the rpn loss
        if self.training:
            assert gt_boxes is not None
            #rpn_cls_score.data: size = [2, 18, 37, 50],2为bs
            #rpn_data: rpnh中特征图中每个点产生的点对应的标签，框等信息，类型为list
            #rpn_data: rpn_data[0]为标签label，大小为9*37*50个
            #首先过滤掉超出图像边界的anchors，对每个标定的ground truth，与其重叠比例IoU最大的anchor记为正样本，这样可以保证每个ground truth至少对应一个正样本anchor
            #对每个anchors，如果其与某个ground truth的重叠比例IoU大于0.7，则记为正样本(目标)；
            #如果小于0.3，则记为负样本(背景)，再从已经得到的正负样本中随机选取256个anchors组成一个minibatch用于训练，而且正负样本的比例为1:1,；如果正样本不够，则补充一些负样本以满足256个anchors用于训练
            rpn_data = self.RPN_anchor_target((rpn_cls_score.data, gt_boxes, im_info, num_boxes))
    
            # compute classification loss
            rpn_cls_score = rpn_cls_score_reshape.permute(0, 2, 3, 1).contiguous().view(batch_size, -1, 2)
            rpn_label = rpn_data[0].view(batch_size, -1)
            
            #去掉label为-1的框,剩下256个框
            rpn_keep = Variable(rpn_label.view(-1).ne(-1).nonzero().view(-1))
            rpn_cls_score = torch.index_select(rpn_cls_score.view(-1,2), 0, rpn_keep)
            rpn_label = torch.index_select(rpn_label.view(-1), 0, rpn_keep.data)
            rpn_label = Variable(rpn_label.long())
            #计算rpn的分类loss
            self.rpn_loss_cls = F.cross_entropy(rpn_cls_score, rpn_label)
            ##rpn_label [256*bs]维的一维列表，值为0，1表示前景后景
            ##rpn_cls_score [256*bs,2]维的二维列表，值为小数表示概率
            
            fg_cnt = torch.sum(rpn_label.data.ne(0))
            
            rpn_bbox_targets, rpn_bbox_inside_weights, rpn_bbox_outside_weights = rpn_data[1:]
            # compute bbox regression loss
            rpn_bbox_inside_weights = Variable(rpn_bbox_inside_weights)
            rpn_bbox_outside_weights = Variable(rpn_bbox_outside_weights)
            rpn_bbox_targets = Variable(rpn_bbox_targets)

            #rpn_bbox_targets为特征图中所有点对应的框，大小为[2, 36, 50, 37]，2为bs
            
            
            #'rpn_box_inside_targets':回归损失函数中的样本权值，正样本为1，负样本为0，相当于损失函数中的p*，损失函数见：https://senitco.github.io/2017/09/02/faster-rcnn/
            #'rpn_box_outside_targets':分类损失函数和回归损失函数的平衡权重，相当于λ
            #'rpn_bbox_targets':anchor与ground truth的回归参数[dx,dy,dw,dh]
            self.rpn_loss_box = _smooth_l1_loss(rpn_bbox_pred, rpn_bbox_targets, rpn_bbox_inside_weights,
                                                            rpn_bbox_outside_weights, sigma=3, dim=[1,2,3])

        return rois, self.rpn_loss_cls, self.rpn_loss_box
