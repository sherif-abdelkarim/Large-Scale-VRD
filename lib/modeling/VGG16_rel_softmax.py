# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import absolute_import
from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division

import logging

from caffe2.python import core as caffe2core
from core.config_rel import cfg
import utils.blob as blob_utils

logger = logging.getLogger(__name__)

dropout_ratio = cfg.TRAIN.DROPOUT


def create_model(model):
    logger.info(' | VGG-16 yall softmax {}'.format(cfg.DATASET))

    model.loss_set = []

    # 1. visual modules
    blob, dim, spatial_scale = add_VGG16_conv5_body(model)

    # sbj and obj always share their branches
    blob_sbj, dim_sbj, spatial_scale_sbj = add_VGG16_roi_fc_head_labeled_shared(
        model, 'sbj', blob, dim, spatial_scale)

    blob_obj, dim_obj, spatial_scale_obj = add_VGG16_roi_fc_head_labeled_shared(
        model, 'obj', blob, dim, spatial_scale)

    blob_rel_prd, dim_rel_prd, \
    blob_rel_sbj, dim_rel_sbj, \
    blob_rel_obj, dim_rel_obj, \
    spatial_scale_rel = add_VGG16_roi_fc_head_rel_spo_late_fusion(
        model, blob, dim, spatial_scale)

    add_visual_embedding(
        model, blob_sbj, dim_sbj, blob_obj, dim_obj,
        blob_rel_prd, dim_rel_prd,
        blob_rel_sbj, dim_rel_sbj,
        blob_rel_obj, dim_rel_obj)

    add_embd_fusion_for_p(model)


    logits_sbj = model.add_FC_layer_with_weight_name('x_sbj_and_obj_out',
        x_blob_sbj, 'logits_sbj', cfg.OUTPUT_EMBEDDING_DIM, cfg.MODEL.NUM_CLASSES_SBJ_OBJ)

    logits_obj = model.add_FC_layer_with_weight_name('x_sbj_and_obj_out',
        x_blob_obj, 'logits_obj', cfg.OUTPUT_EMBEDDING_DIM, cfg.MODEL.NUM_CLASSES_SBJ_OBJ)

    logits_rel = model.FC(x_blob_rel, 'logits_rel',cfg.OUTPUT_EMBEDDING_DIM, cfg.MODEL.NUM_CLASSES_PRD,
                          weight_init=('GaussianFill', {'std': 0.01}),bias_init=('ConstantFill', {'value': 0.}))

    # During testing, get topk labels and scores
    if not model.train:
        add_labels_and_scores_topk(model, 'sbj', logits_sbj)
        add_labels_and_scores_topk(model, 'obj', logits_obj)
        add_labels_and_scores_topk(model, 'rel', logits_rel)

    # # 2. language modules and losses
    if model.train:
        add_embd_pos_neg_splits(model, 'sbj')
        add_embd_pos_neg_splits(model, 'obj')
        add_embd_pos_neg_splits(model, 'rel')

        add_softmax_losses(model, 'sbj')
        add_softmax_losses(model, 'obj')
        add_softmax_losses(model, 'rel')

    loss_gradients = blob_utils.get_loss_gradients(model, model.loss_set)
    model.AddLosses(model.loss_set)
    return loss_gradients if model.train else None


def add_VGG16_conv5_body(model):
    model.Conv('data', 'conv1_1', 3, 64, 3, pad=1, stride=1)
    model.Relu('conv1_1', 'conv1_1')
    model.Conv('conv1_1', 'conv1_2', 64, 64, 3, pad=1, stride=1)
    model.Relu('conv1_2', 'conv1_2')
    model.MaxPool('conv1_2', 'pool1', kernel=2, pad=0, stride=2)
    model.Conv('pool1', 'conv2_1', 64, 128, 3, pad=1, stride=1)
    model.Relu('conv2_1', 'conv2_1')
    model.Conv('conv2_1', 'conv2_2', 128, 128, 3, pad=1, stride=1)
    model.Relu('conv2_2', 'conv2_2')
    model.MaxPool('conv2_2', 'pool2', kernel=2, pad=0, stride=2)
    model.StopGradient('pool2', 'pool2')
    model.Conv('pool2', 'conv3_1', 128, 256, 3, pad=1, stride=1)
    model.Relu('conv3_1', 'conv3_1')
    model.Conv('conv3_1', 'conv3_2', 256, 256, 3, pad=1, stride=1)
    model.Relu('conv3_2', 'conv3_2')
    model.Conv('conv3_2', 'conv3_3', 256, 256, 3, pad=1, stride=1)
    model.Relu('conv3_3', 'conv3_3')
    model.MaxPool('conv3_3', 'pool3', kernel=2, pad=0, stride=2)
    model.Conv('pool3', 'conv4_1', 256, 512, 3, pad=1, stride=1)
    model.Relu('conv4_1', 'conv4_1')
    model.Conv('conv4_1', 'conv4_2', 512, 512, 3, pad=1, stride=1)
    model.Relu('conv4_2', 'conv4_2')
    model.Conv('conv4_2', 'conv4_3', 512, 512, 3, pad=1, stride=1)
    model.Relu('conv4_3', 'conv4_3')
    model.MaxPool('conv4_3', 'pool4', kernel=2, pad=0, stride=2)
    model.Conv('pool4', 'conv5_1', 512, 512, 3, pad=1, stride=1)
    model.Relu('conv5_1', 'conv5_1')
    model.Conv('conv5_1', 'conv5_2', 512, 512, 3, pad=1, stride=1)
    model.Relu('conv5_2', 'conv5_2')
    model.Conv('conv5_2', 'conv5_3', 512, 512, 3, pad=1, stride=1)
    blob_out = model.Relu('conv5_3', 'conv5_3')
    return blob_out, 512, 1. / 16.


def add_VGG16_roi_fc_head_labeled_shared(model, label, blob_in, dim_in, spatial_scale):
    prefix = label + '_'
    model.RoIFeatureTransform(
        blob_in, prefix + 'pool5',
        blob_rois=prefix + 'rois',
        method=cfg.FAST_RCNN.ROI_XFORM_METHOD,
        resolution=7,
        sampling_ratio=cfg.FAST_RCNN.ROI_XFORM_SAMPLING_RATIO,
        spatial_scale=spatial_scale)

    model.add_FC_layer_with_weight_name(
        'fc6', prefix + 'pool5', prefix + 'fc6', dim_in * 7 * 7, 4096)
    model.Relu(prefix + 'fc6', prefix + 'fc6')
    model.Dropout(
        prefix + 'fc6', prefix + 'fc6_dropout',
        ratio=dropout_ratio, is_test=(not model.train))
    model.add_FC_layer_with_weight_name(
        'fc7', prefix + 'fc6_dropout', prefix + 'fc7', 4096, 4096)
    model.Relu(prefix + 'fc7', prefix + 'fc7')
    blob_out = model.Dropout(
        prefix + 'fc7', prefix + 'fc7_dropout',
        ratio=dropout_ratio, is_test=(not model.train))

    return blob_out, 4096, spatial_scale


def add_VGG16_roi_fc_head_rel_spo_late_fusion(
        model, blob_in, dim_in, spatial_scale):
    model.RoIFeatureTransform(
        blob_in, 'rel_roi_pool_sbj',
        blob_rois='rel_rois_sbj',
        method=cfg.FAST_RCNN.ROI_XFORM_METHOD,
        resolution=7,
        sampling_ratio=cfg.FAST_RCNN.ROI_XFORM_SAMPLING_RATIO,
        spatial_scale=spatial_scale)
    model.RoIFeatureTransform(
        blob_in, 'rel_roi_pool_obj',
        blob_rois='rel_rois_obj',
        method=cfg.FAST_RCNN.ROI_XFORM_METHOD,
        resolution=7,
        sampling_ratio=cfg.FAST_RCNN.ROI_XFORM_SAMPLING_RATIO,
        spatial_scale=spatial_scale)
    model.RoIFeatureTransform(
        blob_in, 'rel_roi_pool_prd',
        blob_rois='rel_rois_prd',
        method=cfg.FAST_RCNN.ROI_XFORM_METHOD,
        resolution=7,
        sampling_ratio=cfg.FAST_RCNN.ROI_XFORM_SAMPLING_RATIO,
        spatial_scale=spatial_scale)

    model.add_FC_layer_with_weight_name(
        'fc6', 'rel_roi_pool_sbj', 'rel_sbj_fc6', dim_in * 7 * 7, 4096)
    model.Relu('rel_sbj_fc6', 'rel_sbj_fc6')
    model.Dropout(
        'rel_sbj_fc6', 'rel_sbj_fc6_dropout',
        ratio=dropout_ratio, is_test=(not model.train))
    model.add_FC_layer_with_weight_name(
        'fc7', 'rel_sbj_fc6_dropout', 'rel_sbj_fc7', 4096, 4096)
    model.Relu('rel_sbj_fc7', 'rel_sbj_fc7')
    s_sbj = model.Dropout(
        'rel_sbj_fc7', 'rel_sbj_fc7_dropout',
        ratio=dropout_ratio, is_test=(not model.train))
    if cfg.MODEL.SPECS.find('not_stop_gradient') < 0:
        model.StopGradient(s_sbj, s_sbj)  # this is to stop the gradient from p branch

    model.add_FC_layer_with_weight_name(
        'fc6', 'rel_roi_pool_obj', 'rel_obj_fc6', dim_in * 7 * 7, 4096)
    model.Relu('rel_obj_fc6', 'rel_obj_fc6')
    model.Dropout(
        'rel_obj_fc6', 'rel_obj_fc6_dropout',
        ratio=dropout_ratio, is_test=(not model.train))
    model.add_FC_layer_with_weight_name(
        'fc7', 'rel_obj_fc6_dropout', 'rel_obj_fc7', 4096, 4096)
    model.Relu('rel_obj_fc7', 'rel_obj_fc7')
    s_obj = model.Dropout(
        'rel_obj_fc7', 'rel_obj_fc7_dropout',
        ratio=dropout_ratio, is_test=(not model.train))
    if cfg.MODEL.SPECS.find('not_stop_gradient') < 0:
        model.StopGradient(s_obj, s_obj)  # this is to stop the gradient from p branch

    model.FC(
        'rel_roi_pool_prd', 'rel_fc6_prd', dim_in * 7 * 7, 4096,
        weight_init=('GaussianFill', {'std': 0.01}),
        bias_init=('ConstantFill', {'value': 0.}))
    model.Relu('rel_fc6_prd', 'rel_fc6_prd')
    model.Dropout(
        'rel_fc6_prd', 'rel_fc6_prd_dropout',
        ratio=dropout_ratio, is_test=(not model.train))
    model.FC(
        'rel_fc6_prd_dropout', 'rel_fc7_prd', 4096, 4096,
        weight_init=('GaussianFill', {'std': 0.01}),
        bias_init=('ConstantFill', {'value': 0.}))
    model.Relu('rel_fc7_prd', 'rel_fc7_prd')
    s_prd = model.Dropout(
        'rel_fc7_prd', 'rel_fc7_prd_dropout',
        ratio=dropout_ratio, is_test=(not model.train))

    if cfg.MODEL.TYPE.find('spo_for_p') >= 0:
        s_rel = model.Concat([s_sbj, s_obj, s_prd], 'rel_concat')
        dim_s_rel = 4096 * 3
        return s_rel, dim_s_rel, s_sbj, 4096, s_obj, 4096, spatial_scale
    else:
        return s_prd, 4096, s_sbj, 4096, s_obj, 4096, spatial_scale


# ---------------------------------------------------------------------------- #
# Helper functions for building various re-usable network bits
# ---------------------------------------------------------------------------- #
def add_visual_embedding(model,
                         blob_sbj, dim_sbj,
                         blob_obj, dim_obj,
                         blob_rel_prd, dim_rel_prd,
                         blob_rel_sbj, dim_rel_sbj,
                         blob_rel_obj, dim_rel_obj):
    model.add_FC_layer_with_weight_name(
        'x_sbj_and_obj',
        blob_sbj, 'x_sbj_raw', dim_sbj, cfg.OUTPUT_EMBEDDING_DIM,
        weight_init=('GaussianFill', {'std': 0.01}),
        bias_init=('ConstantFill', {'value': 0.}))
    model.add_FC_layer_with_weight_name(
        'x_sbj_and_obj',
        blob_obj, 'x_obj_raw', dim_obj, cfg.OUTPUT_EMBEDDING_DIM,
        weight_init=('GaussianFill', {'std': 0.01}),
        bias_init=('ConstantFill', {'value': 0.}))
    if cfg.MODEL.SUBTYPE.find('w_ishans') == 0:
        model.FC(
            blob_rel_prd, 'x_rel_prd_raw_1',
            dim_rel_prd, 4 * cfg.OUTPUT_EMBEDDING_DIM,
            weight_init=('GaussianFill', {'std': 0.01}),
            bias_init=('ConstantFill', {'value': 0.}))
        model.LeakyRelu('x_rel_prd_raw_1', 'x_rel_prd_raw_1', alpha=0.1)
        model.FC(
            'x_rel_prd_raw_1', 'x_rel_prd_raw_2',
            4 * cfg.OUTPUT_EMBEDDING_DIM, 2 * cfg.OUTPUT_EMBEDDING_DIM,
            weight_init=('GaussianFill', {'std': 0.01}),
            bias_init=('ConstantFill', {'value': 0.}))
        model.LeakyRelu('x_rel_prd_raw_2', 'x_rel_prd_raw_2', alpha=0.1)
        model.FC(
            'x_rel_prd_raw_2', 'x_rel_prd_raw_3',
            2 * cfg.OUTPUT_EMBEDDING_DIM, cfg.OUTPUT_EMBEDDING_DIM,
            weight_init=('GaussianFill', {'std': 0.01}),
            bias_init=('ConstantFill', {'value': 0.}))
        model.net.Alias('x_rel_prd_raw_3', 'x_rel_prd_raw')
    else:
        model.FC(
            blob_rel_prd, 'x_rel_prd_raw_1',
            dim_rel_prd, cfg.OUTPUT_EMBEDDING_DIM,
            weight_init=('GaussianFill', {'std': 0.01}),
            bias_init=('ConstantFill', {'value': 0.}))
        if cfg.MODEL.SUBTYPE.find('w_ishans') == 0:
            model.Relu('x_rel_prd_raw_1', 'x_rel_prd_raw_1')
            model.FC(
                'x_rel_prd_raw_1', 'x_rel_prd_raw_2',
                cfg.OUTPUT_EMBEDDING_DIM, cfg.OUTPUT_EMBEDDING_DIM,
                weight_init=('GaussianFill', {'std': 0.01}),
                bias_init=('ConstantFill', {'value': 0.}))
            model.net.Alias('x_rel_prd_raw_2', 'x_rel_prd_raw')
        else:
            model.net.Alias('x_rel_prd_raw_1', 'x_rel_prd_raw')
    model.net.Normalize('x_sbj_raw', 'x_sbj')
    model.net.Normalize('x_obj_raw', 'x_obj')
    model.net.Normalize('x_rel_prd_raw', 'x_rel_prd')

    # get x_rel_sbj and x_rel_obj for the p branch
    if model.train and cfg.MODEL.SUBTYPE.find('embd_fusion') >= 0:
        model.add_FC_layer_with_weight_name(
            'x_sbj_and_obj',
            blob_rel_sbj, 'x_rel_sbj_raw',
            dim_rel_sbj, cfg.OUTPUT_EMBEDDING_DIM,
            weight_init=('GaussianFill', {'std': 0.01}),
            bias_init=('ConstantFill', {'value': 0.}))
        model.add_FC_layer_with_weight_name(
            'x_sbj_and_obj',
            blob_rel_obj, 'x_rel_obj_raw',
            dim_rel_obj, cfg.OUTPUT_EMBEDDING_DIM,
            weight_init=('GaussianFill', {'std': 0.01}),
            bias_init=('ConstantFill', {'value': 0.}))
        x_rel_sbj = model.net.Normalize('x_rel_sbj_raw', 'x_rel_sbj')
        x_rel_obj = model.net.Normalize('x_rel_obj_raw', 'x_rel_obj')
        if cfg.MODEL.SPECS.find('not_stop_gradient') < 0:
            # this is to stop gradients from x_rel_sbj and x_rel_obj
            model.StopGradient(x_rel_sbj, x_rel_sbj)
            model.StopGradient(x_rel_obj, x_rel_obj)


def add_embd_fusion_for_p(model):
    if cfg.MODEL.SUBTYPE.find('embd_fusion') < 0:
        model.net.Alias('x_rel_prd', 'x_rel_raw_final')
    else:
        if model.train:
            x_spo = model.Concat(
                ['x_rel_sbj', 'x_rel_obj', 'x_rel_prd'], 'x_spo')
            dim_x_spo = cfg.OUTPUT_EMBEDDING_DIM * 3
        else:
            x_spo = model.Concat(
                ['x_sbj', 'x_obj', 'x_rel_prd'], 'x_spo')
            dim_x_spo = cfg.OUTPUT_EMBEDDING_DIM * 3
        if cfg.MODEL.SUBTYPE.find('w_ishans') >= 0:
            model.FC(
                x_spo, 'x_rel_raw',
                dim_x_spo, 4 * cfg.OUTPUT_EMBEDDING_DIM,
                weight_init=('GaussianFill', {'std': 0.01}),
                bias_init=('ConstantFill', {'value': 0.}))
            model.LeakyRelu('x_rel_raw', 'x_rel_raw', alpha=0.1)
            model.FC(
                'x_rel_raw', 'x_rel_raw_2',
                4 * cfg.OUTPUT_EMBEDDING_DIM, 2 * cfg.OUTPUT_EMBEDDING_DIM,
                weight_init=('GaussianFill', {'std': 0.01}),
                bias_init=('ConstantFill', {'value': 0.}))
            model.LeakyRelu('x_rel_raw_2', 'x_rel_raw_2', alpha=0.1)
            model.FC(
                'x_rel_raw_2', 'x_rel_raw_3',
                2 * cfg.OUTPUT_EMBEDDING_DIM, cfg.OUTPUT_EMBEDDING_DIM,
                weight_init=('GaussianFill', {'std': 0.01}),
                bias_init=('ConstantFill', {'value': 0.}))
            model.net.Alias('x_rel_raw_3', 'x_rel_raw_final')
        else:
            model.FC(
                x_spo, 'x_rel_raw',
                dim_x_spo, cfg.OUTPUT_EMBEDDING_DIM,
                weight_init=('GaussianFill', {'std': 0.01}),
                bias_init=('ConstantFill', {'value': 0.}))
            if cfg.MODEL.SUBTYPE.find('w_relu') >= 0:
                model.Relu('x_rel_raw', 'x_rel_raw')
                model.FC(
                    'x_rel_raw', 'x_rel_raw_2',
                    cfg.OUTPUT_EMBEDDING_DIM, cfg.OUTPUT_EMBEDDING_DIM,
                    weight_init=('GaussianFill', {'std': 0.01}),
                    bias_init=('ConstantFill', {'value': 0.}))
                model.net.Alias('x_rel_raw_2', 'x_rel_raw_final')
            else:
                model.net.Alias('x_rel_raw', 'x_rel_raw_final')
    model.net.Normalize('x_rel_raw_final', 'x_rel')


def add_embd_pos_neg_splits(model, label, sublabel=''):
    preprefix = label + '_'
    if sublabel == '':
        prefix = preprefix
        suffix = '_' + label
    else:
        prefix = preprefix + sublabel + '_'
        suffix = '_' + label + '_' + sublabel

    if cfg.MODEL.SUBTYPE.find('xp_only') < 0:
        model.net.Slice(['logits' + suffix, preprefix + 'pos_starts',
                         preprefix + 'pos_ends'], 'xp' + suffix)
        model.net.Alias('xp' + suffix, 'scaled_xp' + suffix)
        # model.Scale('xp' + suffix, 'scaled_xp' + suffix, scale=cfg.TRAIN.NORM_SCALAR)
        if suffix == '_rel':
            model.net.Slice(['x_rel_raw_final', prefix + 'pos_starts',
                             prefix + 'pos_ends'], 'xp_rel_raw_final')
        else:
            model.net.Slice(['x_' + label + '_raw', prefix + 'pos_starts',
                             prefix + 'pos_ends'], 'xp_' + label + '_raw')
    else:
        model.net.Alias('logits' + suffix, 'scaled_xp' + suffix)

def add_softmax_losses(model, label):
    prefix = label + '_'
    suffix = '_' + label

    if cfg.MODEL.WEAK_LABELS:
        _, loss_xp_yall = model.net.SoftmaxWithLoss(
            ['scaled_xp' + suffix, prefix + 'pos_labels_float32_w'],
            ['xp_yall_prob' + suffix, 'loss_xp_yall' + suffix],
            scale=1. / cfg.NUM_DEVICES,
            label_prob=1)
        model.loss_set.extend([loss_xp_yall])
    else:
        _, loss_xp_yall = model.net.SoftmaxWithLoss(
            ['scaled_xp' + suffix, prefix + 'pos_labels_int32'],
            ['xp_yall_prob' + suffix, 'loss_xp_yall' + suffix],
            scale=1. / cfg.NUM_DEVICES)

        model.loss_set.extend([loss_xp_yall])


def add_labels_and_scores_topk(model, label, logits):
    suffix = '_' + label
    model.net.TopK(logits, ['scores' + suffix, 'labels' + suffix], k=250)
