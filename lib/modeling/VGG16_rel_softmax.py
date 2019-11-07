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
from caffe2.python import workspace
from core.config_rel import cfg
import utils.blob as blob_utils
import pickle
import math

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

    if cfg.MODEL.MEMORY_MODULE_SBJ_OBJ:
        model.add_centroids_blob_with_weight_name('centroids_obj', cfg.MODEL.NUM_CLASSES_SBJ_OBJ, cfg.OUTPUT_EMBEDDING_DIM)
        #if cfg.DEBUG:
        #    if cfg.MODEL.SUBTYPE.find('xp_only') < 0:
        #        model.Scale('centroids_obj', 'centroids_obj', scale=cfg.TRAIN.NORM_SCALAR)
        model.net.Alias('centroids_obj', 'centroids_sbj')
        std = 1. / math.sqrt(cfg.OUTPUT_EMBEDDING_DIM)
        model.add_weight_blob_with_weight_name('weight_obj', cfg.MODEL.NUM_CLASSES_SBJ_OBJ, cfg.OUTPUT_EMBEDDING_DIM, -std, std)
        model.net.Alias('weight_obj', 'weight_sbj')

    if cfg.MODEL.MEMORY_MODULE_PRD:
        model.add_centroids_blob_with_weight_name('centroids_rel', cfg.MODEL.NUM_CLASSES_PRD, cfg.OUTPUT_EMBEDDING_DIM)
        #if cfg.DEBUG:
        #    if cfg.MODEL.SUBTYPE.find('xp_only') < 0:
        #        model.Scale('centroids_rel', 'centroids_rel', scale=cfg.TRAIN.NORM_SCALAR)

        std = 1. / math.sqrt(cfg.OUTPUT_EMBEDDING_DIM)
        model.add_weight_blob_with_weight_name('weight_rel', cfg.MODEL.NUM_CLASSES_PRD, cfg.OUTPUT_EMBEDDING_DIM, -std, std)

    add_visual_embedding(
        model, blob_sbj, dim_sbj, blob_obj, dim_obj,
        blob_rel_prd, dim_rel_prd,
        blob_rel_sbj, dim_rel_sbj,
        blob_rel_obj, dim_rel_obj)

    add_embd_fusion_for_p(model)

    model.net.ConstantFill([], 'one_blob', shape=[1], value=1.0)
    model.net.ConstantFill([], 'scale_blob', shape=[1], value=16.0)

    if model.train:
        add_embd_pos_neg_splits(model, 'sbj')
        add_embd_pos_neg_splits(model, 'obj')
        add_embd_pos_neg_splits(model, 'rel')
    else:
        model.net.Alias('x_sbj', 'scaled_xp_sbj')
        model.net.Alias('x_obj', 'scaled_xp_obj')
        model.net.Alias('x_rel', 'scaled_xp_rel')

    x_blob_sbj = 'scaled_xp_sbj'
    x_blob_obj = 'scaled_xp_obj'
    x_blob_rel = 'scaled_xp_rel'

    if cfg.MODEL.MEMORY_MODULE_SBJ_OBJ or cfg.MODEL.MEMORY_MODULE_PRD:
        model.StopGradient(x_blob_sbj, x_blob_sbj)
        model.StopGradient(x_blob_obj, x_blob_obj)
        model.StopGradient(x_blob_rel, x_blob_rel)

    if cfg.MODEL.MEMORY_MODULE_SBJ_OBJ or cfg.MODEL.MEMORY_MODULE_PRD:

        model.net.Shape(x_blob_sbj, 'x_sbj_shape')
        model.net.Shape(x_blob_obj, 'x_obj_shape')
        model.net.Shape(x_blob_rel, 'x_rel_shape')
        model.net.Slice(['x_sbj_shape'], 'batch_size_sbj', starts=[0], ends=[1])
        model.net.Slice(['x_obj_shape'], 'batch_size_obj', starts=[0], ends=[1])
        model.net.Slice(['x_rel_shape'], 'batch_size_rel', starts=[0], ends=[1])
        model.net.Slice([x_blob_sbj], 'single_row_sbj', starts=[0, 0], ends=[-1, 1])
        model.net.Slice([x_blob_obj], 'single_row_obj', starts=[0, 0], ends=[-1, 1])
        model.net.Slice([x_blob_rel], 'single_row_rel', starts=[0, 0], ends=[-1, 1])

        model.net.ConstantFill(['single_row_sbj'], 'scale_10_blob_sbj', value=10.0)
        model.net.ConstantFill(['single_row_obj'], 'scale_10_blob_obj', value=10.0)
        model.net.ConstantFill(['single_row_rel'], 'scale_10_blob_rel', value=10.0)

        model.net.ConstantFill([], 'neg_two_blob', shape=[1], value=-2.0)
        model.net.ConstantFill([], 'neg_one_blob', shape=[1], value=-1.0)
        model.net.ConstantFill([x_blob_sbj], 'zero_blob_x_sbj', value=0.0)
        model.net.ConstantFill([x_blob_obj], 'zero_blob_x_obj', value=0.0)
        model.net.ConstantFill([x_blob_rel], 'zero_blob_x_rel', value=0.0)

        model.net.ConstantFill([x_blob_sbj], 'one_blob_x_sbj', value=1.0)
        model.net.ConstantFill([x_blob_obj], 'one_blob_x_obj', value=1.0)
        model.net.ConstantFill([x_blob_rel], 'one_blob_x_rel', value=1.0)

        model.net.ConstantFill([], 'zero_blob_c_sbj', shape=[cfg.MODEL.NUM_CLASSES_SBJ_OBJ, cfg.OUTPUT_EMBEDDING_DIM],
                               value=0.0)
        model.net.ConstantFill([], 'zero_blob_c_obj', shape=[cfg.MODEL.NUM_CLASSES_SBJ_OBJ, cfg.OUTPUT_EMBEDDING_DIM],
                               value=0.0)
        model.net.ConstantFill([], 'zero_blob_c_rel', shape=[cfg.MODEL.NUM_CLASSES_PRD, cfg.OUTPUT_EMBEDDING_DIM],
                               value=0.0)

        model.net.ConstantFill([], 'one_blob_c_sbj', shape=[cfg.MODEL.NUM_CLASSES_SBJ_OBJ, cfg.OUTPUT_EMBEDDING_DIM],
                               value=1.0)
        model.net.ConstantFill([], 'one_blob_c_obj', shape=[cfg.MODEL.NUM_CLASSES_SBJ_OBJ, cfg.OUTPUT_EMBEDDING_DIM],
                               value=1.0)
        model.net.ConstantFill([], 'one_blob_c_rel', shape=[cfg.MODEL.NUM_CLASSES_PRD, cfg.OUTPUT_EMBEDDING_DIM],
                               value=1.0)

        model.net.ConstantFill([], 'num_classes_sbj', shape=[1, 0], value=cfg.MODEL.NUM_CLASSES_SBJ_OBJ)
        model.net.ConstantFill([], 'num_classes_obj', shape=[1, 0], value=cfg.MODEL.NUM_CLASSES_SBJ_OBJ)
        model.net.ConstantFill([], 'num_classes_rel', shape=[1, 0], value=cfg.MODEL.NUM_CLASSES_PRD)

        if cfg.MODEL.MEMORY_MODULE_SBJ_OBJ:
            add_memory_module(model, x_blob_sbj, 'centroids_obj', 'sbj', cfg.MODEL.NUM_CLASSES_SBJ_OBJ)
            add_memory_module(model, x_blob_obj, 'centroids_obj', 'obj', cfg.MODEL.NUM_CLASSES_SBJ_OBJ)
        else:
            model.add_FC_layer_with_weight_name(
                'x_sbj_and_obj_out',
                x_blob_sbj, 'logits_sbj', cfg.OUTPUT_EMBEDDING_DIM, cfg.MODEL.NUM_CLASSES_SBJ_OBJ)

            model.add_FC_layer_with_weight_name(
                'x_sbj_and_obj_out',
                x_blob_obj, 'logits_obj', cfg.OUTPUT_EMBEDDING_DIM, cfg.MODEL.NUM_CLASSES_SBJ_OBJ)

        if cfg.MODEL.MEMORY_MODULE_PRD:
            add_memory_module(model, x_blob_rel, 'centroids_rel', 'rel', cfg.MODEL.NUM_CLASSES_PRD)
        else:
            model.FC(
                x_blob_rel, 'logits_rel',
                cfg.OUTPUT_EMBEDDING_DIM, cfg.MODEL.NUM_CLASSES_PRD,
                weight_init=('GaussianFill', {'std': 0.01}),
                bias_init=('ConstantFill', {'value': 0.}))


    else:
        model.add_FC_layer_with_weight_name(
            'x_sbj_and_obj_out',
            x_blob_sbj, 'logits_sbj', cfg.OUTPUT_EMBEDDING_DIM, cfg.MODEL.NUM_CLASSES_SBJ_OBJ)

        model.add_FC_layer_with_weight_name(
            'x_sbj_and_obj_out',
            x_blob_obj, 'logits_obj', cfg.OUTPUT_EMBEDDING_DIM, cfg.MODEL.NUM_CLASSES_SBJ_OBJ)

        model.FC(
            x_blob_rel, 'logits_rel',
            cfg.OUTPUT_EMBEDDING_DIM, cfg.MODEL.NUM_CLASSES_PRD,
            weight_init=('GaussianFill', {'std': 0.01}),
            bias_init=('ConstantFill', {'value': 0.}))

    # During testing, get topk labels and scores
    if not model.train:
        add_labels_and_scores_topk(model, 'sbj')
        add_labels_and_scores_topk(model, 'obj')
        add_labels_and_scores_topk(model, 'rel')

    # # 2. language modules and losses
    if model.train:
        add_softmax_losses(model, 'sbj')
        add_softmax_losses(model, 'obj')
        add_softmax_losses(model, 'rel')
        if cfg.MODEL.MEMORY_MODULE_SBJ_OBJ:
            add_centroids_loss(model, x_blob_sbj, 'sbj', cfg.MODEL.NUM_CLASSES_SBJ_OBJ, 'num_classes_sbj')
            add_centroids_loss(model, x_blob_obj, 'obj', cfg.MODEL.NUM_CLASSES_SBJ_OBJ, 'num_classes_obj')

        if cfg.MODEL.MEMORY_MODULE_PRD:
            add_centroids_loss(model, x_blob_rel, 'rel', cfg.MODEL.NUM_CLASSES_PRD, 'num_classes_rel')

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
        model.net.Slice(['x' + suffix, preprefix + 'pos_starts',
                         preprefix + 'pos_ends'], 'xp' + suffix)
        model.Scale('xp' + suffix, 'scaled_xp' + suffix, scale=cfg.TRAIN.NORM_SCALAR)
        if suffix == '_rel':
            model.net.Slice(['x_rel_raw_final', prefix + 'pos_starts',
                             prefix + 'pos_ends'], 'xp_rel_raw_final')
        else:
            model.net.Slice(['x_' + label + '_raw', prefix + 'pos_starts',
                             prefix + 'pos_ends'], 'xp_' + label + '_raw')
    else:
        model.net.Alias('x' + suffix, 'scaled_xp' + suffix)


def add_softmax_losses(model, label):
    prefix = label + '_'
    suffix = '_' + label

    if cfg.MODEL.WEAK_LABELS:
        _, loss_xp_yall = model.net.SoftmaxWithLoss(
            ['logits' + suffix, prefix + 'pos_labels_float32_w'],
            ['xp_yall_prob' + suffix, 'loss_xp_yall' + suffix],
            scale=1. / cfg.NUM_DEVICES,
            label_prob=1)
        model.loss_set.extend([loss_xp_yall])
    else:
        _, loss_xp_yall = model.net.SoftmaxWithLoss(
        ['logits' + suffix, prefix + 'pos_labels_int32'],
        ['xp_yall_prob' + suffix, 'loss_xp_yall' + suffix],
        scale=1. / cfg.NUM_DEVICES)

        model.loss_set.extend([loss_xp_yall])


def add_centroids_loss(model, feat, label, num_classes, num_classes_blob):
    prefix = label + '_'
    suffix = '_' + label

    # batch_size = feat.size(0)

    # calculate attracting loss

    # feat = feat.view(batch_size, -1)
    # (128, 1024)
    # model.net.Reshape([feat],
    #                  ['feat_reshaped' + suffix, 'feat_old_shape' + suffix],
    #                  shape=(batch_size, -1))

    model.net.Alias(feat, 'feat_reshaped' + suffix)
    # To check the dim of centroids and features
    # if feat.size(1) != self.feat_dim:
    #     raise ValueError("Center's dim: {0} should be equal to input feature's \
    #                     dim: {1}".format(self.feat_dim, feat.size(1)))
    # batch_size_tensor = feat.new_empty(1).fill_(batch_size if self.size_average else 1)
    model.net.ConstantFill(['feat_reshaped' + suffix], 'batch_size_tensor' + suffix, value=1.0)

    # loss_attract = self.disccentroidslossfunc(feat, label, self.centroids, batch_size_tensor).squeeze()

    loss_attract = disc_centroids_loss_func(model, 'feat_reshaped' + suffix, prefix + 'pos_labels_int32', 'centroids' + suffix,
                                            'batch_size_tensor' + suffix, label) #TODO: Check the backward function in original implementation
    # calculate repelling loss

    # distmat = torch.pow(feat, 2).sum(dim=1, keepdim=True).expand(batch_size, self.num_classes) + \
    #           torch.pow(self.centroids, 2).sum(dim=1, keepdim=True).expand(self.num_classes, batch_size).t()

    # torch.pow(feat, 2).sum(dim=1, keepdim=True).expand(batch_size, self.num_classes)
    #torch.pow(feat, 2)
    model.net.Sqr(['feat_reshaped' + suffix], ['feat_squared' + suffix])

    #.sum(dim=1, keepdim=True)
    # split = tuple([1 for i in range(cfg.OUTPUT_EMBEDDING_DIM)])
    # tensors_list_names = ['sum_' + str(i) + suffix for i in range(cfg.OUTPUT_EMBEDDING_DIM)]
    # tensors_list = model.net.Split('feat_squared' + suffix, tensors_list_names, axis=1, split=split)
    # model.net.Sum(tensors_list, 'feat_squared_sum' + suffix) #(128, 1)

    model.net.ReduceBackSum('feat_squared' + suffix, 'feat_squared_sum_temp' + suffix, num_reduce_dims=1)
    model.net.ExpandDims('feat_squared_sum_temp' + suffix, 'feat_squared_sum' + suffix, dims=[1])

    #.expand(batch_size, self.num_classes)
    model.net.Tile('feat_squared_sum' + suffix,
                  'feat_squared_sum_expand' + suffix,
                  tiles=num_classes,
                  axis=1) #(128, 1703)

    # torch.pow(self.centroids, 2).sum(dim=1, keepdim=True).expand(self.num_classes, batch_size).t()
    # torch.pow(self.centroids, 2)
    # (1703, 1024)
    model.net.Sqr(['centroids' + suffix], ['centroids_squared' + suffix])

    # #.sum(dim=1, keepdim=True)
    # tensors_list_names = ['centroids_sum_' + str(i) + suffix for i in range(cfg.OUTPUT_EMBEDDING_DIM)]
    # tensors_list = model.net.Split('centroids_squared' + suffix, tensors_list_names, axis=1, split=split)
    # model.net.Sum(tensors_list, 'centroids_squared_sum' + suffix) #(1703, 1)

    model.net.ReduceBackSum('centroids_squared' + suffix, 'centroids_squared_sum_temp' + suffix, num_reduce_dims=1)
    model.net.ExpandDims('centroids_squared_sum_temp' + suffix, 'centroids_squared_sum' + suffix, dims=[1])

    model.Transpose(['centroids_squared_sum' + suffix], ['centroids_squared_sum_T' + suffix]) #(1, 1703)
    model.net.Tile(['centroids_squared_sum_T' + suffix, 'batch_size' + suffix],
                   'centroids_squared_sum_expand_T' + suffix,
                   axis=0) # (128, 1703)

    # distmat = torch.pow(feat, 2).sum(dim=1, keepdim=True).expand(batch_size, self.num_classes) + \
    #           torch.pow(self.centroids, 2).sum(dim=1, keepdim=True).expand(self.num_classes, batch_size).t()

    model.net.Sum(['feat_squared_sum_expand' + suffix, 'centroids_squared_sum_expand_T' + suffix], 'distmat' + suffix) # (128, 1703)


    # feat: (128, 1024)
    # centroid: (1703, 1024)
    # centroid.t(): (1024, 1703)
    # feat * centroid.t(): (128, 1703)
    # distmat.addmm_(1, -2, feat, self.centroids.t())
    model.net.MatMul([feat, 'centroids' + suffix], 'feat_dot_centroids' + suffix, trans_b=1) # (128, 1703)
    model.net.Mul(['feat_dot_centroids' + suffix, 'neg_two_blob'], 'neg_2_feat_dot_centroids' + suffix, broadcast=1)
    model.net.Sum(['distmat' + suffix, 'neg_2_feat_dot_centroids' + suffix], 'distmat_plus_neg_2feat_dot_centroids' + suffix) # (128, 1703)

    # classes = torch.arange(self.num_classes).long().cuda()
    # should produce  'classes' + suffix

    # labels_expand = label.unsqueeze(1).expand(batch_size, self.num_classes)
    #label.unsqueeze(1)
    #model.net.ExpandDims([prefix + 'pos_labels_int32'],
    #                    ['labels_expand' + suffix],
    #                    dims=[1])
    #.expand(batch_size, self.num_classes)
    #model.net.Tile('labels_expand' + suffix,
    #              'labels_expand_tile' + suffix,
    #              tiles=num_classes,
    #              axis=1) #(128, 1703)

    #model.net.ConstantFill(['labels_expand_tile' + suffix], 'zero_blob_mask' + suffix, value=0.0)

    # classes.expand(batch_size, self.num_classes)
    #model.net.ExpandDims(['classes' + suffix],
    #                    ['classes_expand' + suffix],
    #                    dims=[0]) # (1, 1703)
    #model.net.Tile(['classes_expand' + suffix, 'batch_size' + suffix],
    #               'classes_expand_tile' + suffix,
    #               axis=0) # (128, 1703)

    # mask = labels_expand.eq(classes.expand(batch_size, self.num_classes))
    # model.net.EQ(['labels_expand_tile' + suffix, 'classes_expand_tile' + suffix], 'mask' + suffix)
    # model.net.OneHot([prefix + 'pos_labels_int32', num_classes_blob], 'mask' + suffix)
    # model.net.ConstantFill([prefix + 'pos_labels_one_hot'], 'ones_mask' + suffix, value=1.0)
    # model.net.Sub(['ones_mask' + suffix, prefix + 'pos_labels_one_hot'], 'neg_mask' + suffix)


    # distmat_neg = distmat

    # distmat_neg[mask] = 0.0
    #model.net.Where(['mask' + suffix, 'distmat_plus_neg_2feat_dot_centroids' + suffix, 'zero_blob_mask' + suffix], 'distmat_neg' + suffix) # (128, 1703)
    #model.net.Where(['mask' + suffix, 'zero_blob_mask' + suffix, 'distmat_plus_neg_2feat_dot_centroids' + suffix], 'distmat_neg' + suffix) # (128, 1703)
    model.net.Mul(['distmat_plus_neg_2feat_dot_centroids' + suffix, prefix + 'pos_labels_neg_one_hot'], 'distmat_neg' + suffix)
    # margin = 50.0
    margin = 10.0

    #              <------------------------------Input---------------------------------->,  min, max
    # loss_repel = torch.clamp(margin - distmat_neg.sum() / (batch_size * self.num_classes), 0.0, 1e6)
    # distmat_neg.sum() / (batch_size * self.num_classes
    model.net.ReduceBackSum('distmat_neg' + suffix, 'distmat_neg_sum1' + suffix, num_reduce_dims=1)
    model.net.ReduceBackSum('distmat_neg_sum1' + suffix, 'distmat_neg_sum' + suffix, num_reduce_dims=1)
    model.net.ConstantFill(['distmat_neg_sum' + suffix], 'margin_blob' + suffix, value=margin)

    # margin - distmat_neg.sum() / (batch_size * self.num_classes
    model.net.Sub(['margin_blob' + suffix, 'distmat_neg_sum' + suffix], 'margin_minus_distmat_neg' + suffix)

    # torch.clamp()
    model.net.Clip('margin_minus_distmat_neg' + suffix, 'loss_repel' + suffix, min=0.0, max=1e6)
    ## loss = loss_attract + 0.05 * loss_repel

    # loss = loss_attract + 0.01 * loss_repel
    # 0.01 * loss_repel
    model.net.Scale('loss_repel' + suffix, 'loss_repel_scaled' + suffix, scale=0.01)
    if cfg.DEBUG:
        pass
        #model.net.Print('centroids' + suffix, [])
        #model.net.Print('weight' + suffix, [])
        #model.net.Print('distmat_neg' + suffix, [])
        #model.net.Print('reduce_min' + suffix, [])

        #model.net.Print('min_dis_rel', [])
        #model.net.Print('min_dis_test_rel', [])
        #model.net.Print('min_dis_test2_rel', [])
        #model.net.Print('neg_dist_cur_max_rel', [])

        #model.net.Print('neg_dist_cur_rel', [])
        #model.net.Print('dist_cur_rel', [])
        #model.net.Print('dist_cur' + suffix, [])

        #model.net.Slice(['dist_cur_rel'], 'dist_cur_slice_rel', starts=[0, 0], ends=[1, -1])
        #model.net.Print('dist_cur_slice_rel', [])
        #model.net.Slice(['neg_dist_cur_rel'], 'neg_dist_cur_slice_rel', starts=[0, 0], ends=[1, -1])
        #model.net.Print('neg_dist_cur_slice_rel', [])

        #model.net.Slice(['dist_cur' + suffix], 'dist_cur' + '_slice' + suffix, starts=[0, 0], ends=[5, 5])
        #model.net.Print('distmat' + suffix, [])
        #model.net.Print(feat, [])
        #model.net.Print(model.net.Shape('distmat_neg_sum' + suffix, 'distmat_neg_sum' + suffix + '_shape'), [])
        #model.net.Print(model.net.Shape('loss_repel_scaled' + suffix, 'loss_repel_scaled' + suffix + '_shape'), [])
        #model.net.Print(model.net.Shape(loss_attract, 'loss_attract' + suffix + '_shape'), [])

    # loss_attract + 0.01 * loss_repel
    loss_large_margin = model.net.Sum([loss_attract, 'loss_repel_scaled' + suffix], 'loss_large_margin' + suffix)
    model.loss_set.extend([loss_large_margin]) 
    #model.net.Print(loss_large_margin, [])


def disc_centroids_loss_func(model, feature, labels, centroids_blob_name, batch_size_tensor, label):
    prefix = label + '_'
    suffix = '_' + label

    # centroids_expand = centroids.unsqueeze(0).expand(batch_size, -1, -1)
    #model.net.ExpandDims([centroids_blob_name],
    #                    ['centroids_expanddims' + suffix],
    #                    dims=[0])

    #model.net.Tile(['centroids_expanddims' + suffix, batch_size_tensor],
    #              'centroids_expand' + suffix,
    #              axis=0)


    # ctx.save_for_backward(feature, label, centroids, batch_size)
    # centroids_batch = centroids.index_select(0, label.long())
    # centroids: (1703, 1024)
    # centroids.index_select: (128, 1024)
    # feature: (128, 1024)
    model.net.BatchGather([model.net.Transpose(['centroids' + suffix], ['centroids_T' + suffix]), prefix + 'pos_labels_int32'], 'centroids_batch_T' + suffix)
    model.net.Transpose(['centroids_batch_T' + suffix], ['centroids_batch' + suffix])
    #model.net.Print(model.net.Shape('centroids_batch' + suffix, 'centroids_batch' + suffix + '_shape'), [])
    #model.net.Print(model.net.Shape('centroids' + suffix, 'centroids' + suffix + '_shape'), [])
    #model.net.Print(model.net.Shape(feature, 'feature' + suffix + '_shape'), [])
    #model.net.Print(prefix + 'pos_labels_int32', [])

    # return (feature - centroids_batch).pow(2).sum() / 2.0 / batch_size
    # (feature - centroids_batch)
    # out shape: (128, 1024)
    #model.net.Sub([feature, 'centroids_batch' + suffix], 'feature_minus_centroids' + suffix)

    # (feature - centroids_batch).pow(2)
    # out shape: (128, 1024)
    #model.net.Sqr('feature_minus_centroids' + suffix, 'feature_minus_centroids_squared' + suffix)

    # (feature - centroids_batch).pow(2).sum()
    # out shape: (1,)
    #model.net.ReduceBackSum(['feature_minus_centroids_squared' + suffix],
    #                        'feature_minus_centroids_squared_sum' + suffix,
    #                        num_reduce_dims=2)

    # (feature - centroids_batch).pow(2).sum() / 2.0
    #model.net.Scale('feature_minus_centroids_squared_sum' + suffix,
    #                'feature_minus_centroids_squared_sum_scaled' + suffix,
    #                scale=0.5)
    model.net.SquaredL2Distance([feature, 'centroids_batch' + suffix], 'feat_centroid_l2_dist' + suffix)
    model.net.ReduceBackSum(['feat_centroid_l2_dist' + suffix],
                            'feat_centroid_l2_dist_sum' + suffix,
                            num_reduce_dims=1)

    #model.net.Print('feat_centroid_l2_dist' + suffix, [])
    #model.net.Print('feat_centroid_l2_dist_sum' + suffix, [])
    #model.net.Print(model.net.Shape('feat_centroid_l2_dist' + suffix, 'feat_centroid_l2_dist' + suffix + '_shape'), [])
    #loss_attract = model.net.Div(['feat_centroid_l2_dist' + suffix, batch_size_tensor], 'loss_attract' + suffix)
    loss_attract = model.net.Alias('feat_centroid_l2_dist_sum' + suffix, 'loss_attract' + suffix)
    return loss_attract


def add_labels_and_scores_topk(model, label):
    suffix = '_' + label
    model.net.TopK('logits' + suffix, ['scores' + suffix, 'labels' + suffix], k=250)


def add_memory_module(model, x_blob, centroids_blob_name, label, num_classes):
    prefix = label + '_'
    suffix = '_' + label

    # storing direct feature
    direct_feature = x_blob

    # batch_size = cfg.TRAIN.BATCH_SIZE_PER_IM
    feat_size = cfg.OUTPUT_EMBEDDING_DIM

    # set up visual memory
    # x_expand = x.unsqueeze(1).expand(-1, self.num_classes, -1)
    model.net.ExpandDims([x_blob],
                        ['x_expanddims' + suffix],
                        dims=[1])

    model.net.Tile('x_expanddims' + suffix,
                  'x_expand' + suffix,
                  tiles=num_classes,
                  axis=1)

    # centroids_expand = centroids.unsqueeze(0).expand(batch_size, -1, -1)
    model.net.ExpandDims([centroids_blob_name],
                        ['centroids_expanddims' + suffix],
                        dims=[0])

    model.net.Tile(['centroids_expanddims' + suffix, 'batch_size' + suffix],
                  'centroids_expand' + suffix,
                  # tiles=batch_size,
                  axis=0)
    keys_memory = centroids_blob_name

    model.net.Sub(['x_expand' + suffix, 'centroids_expand' + suffix], 'x_minus_c' + suffix)

    #model.net.ConstantFill(['x_minus_c' + suffix],  'one_blob_x_minus_c' + suffix, value=1.0)

    dist_cur = l2_norm(model, 'x_minus_c' + suffix, keepdims=False)
    model.net.Alias(dist_cur, 'dist_cur' + suffix)

    # computing reachability

    if cfg.DEBUG:
        pass

    model.net.ConstantFill([dist_cur], 'debug_blob' + suffix, value=1.0)
    model.net.Div(['debug_blob' + suffix, dist_cur], 'flipped_dist' + suffix)
    #model.net.Scale([dist_cur], 'neg_dist_cur' + suffix, scale=-1.)
    #model.net.ReduceBackMax('neg_dist_cur' + suffix, 'neg_dist_cur_max' + suffix)
    model.net.ReduceBackMax('flipped_dist' + suffix, 'flipped_dist_max' + suffix)
    model.net.ConstantFill(['flipped_dist_max' + suffix], 'debug_blob2' + suffix, value=1.0)
    model.net.Div(['debug_blob2' + suffix, 'flipped_dist_max' + suffix], 'min_dis_temp' + suffix)
    model.net.ExpandDims('min_dis_temp' + suffix, 'min_dis' + suffix, dims=[1])
    #model.net.Scale(['neg_dist_cur_max' + suffix], 'min_dis_test' + suffix, scale=-1.)

    # split = tuple([1 for i in range(num_classes)])
    # tensors_list_names = ['tensor' + str(i) + suffix for i in range(num_classes)]
    # tensors_list = model.net.Split(dist_cur, tensors_list_names, axis=1, split=split)
    #
    # model.net.Min(tensors_list,
    #               'min_dis' + suffix)

    model.net.Div(['scale_10_blob' + suffix, 'min_dis' + suffix], 'scale_over_values' + suffix)

    reachability = model.net.Tile('scale_over_values' + suffix,
                                  'reachability' + suffix,
                                  tiles=feat_size,
                                  axis=1)
    # computing memory feature by querying and associating visual memory
    if suffix == 'rel' and cfg.MODEL.MEMORY_MSG_PASSING:
        fused_memory_features = model.net.Concat([x_blob, 'concept_selector_sbj', 'concept_selector_obj'], axis=1)
        values_memory = add_hallucinator(model, fused_memory_features, 'values_memory' + suffix, feat_size, num_classes)
    else:
        # values_memory = self.fc_hallucinator(x)
        values_memory = add_hallucinator(model, x_blob, 'values_memory' + suffix, feat_size, num_classes)
    # values_memory = values_memory.softmax(dim=1)
    values_memory = model.net.Softmax(values_memory, axis=1)
    # memory_feature = torch.matmul(values_memory, keys_memory)
    memory_feature = model.net.MatMul([values_memory, keys_memory],
                                      'memory_feature' + suffix)

    # computing concept selector
    # concept_selector = self.fc_selector(x)
    if suffix == 'rel' and cfg.MODEL.MEMORY_MSG_PASSING:
        concept_selector = add_selector(model, fused_memory_features, 'concept_selector' + suffix, feat_size)
    else:
        concept_selector = add_selector(model, x_blob, 'concept_selector' + suffix, feat_size)

    # concept_selector = concept_selector.tanh()
    concept_selector = model.net.Tanh(concept_selector)
    # x = reachability * (direct_feature + concept_selector * memory_feature)
    model.net.Mul([concept_selector, memory_feature],
                  'matmul_concep_memory' + suffix)
    model.net.Add([direct_feature, 'matmul_concep_memory' + suffix], 'add_matmul_conc_mem' + suffix)
    #model.net.Print(model.net.Shape('centroids' + suffix, 'centroids' + suffix + '_shape'), [])
    #model.net.Print('centroids' + suffix, [])
    x_out = model.net.Mul([reachability, 'add_matmul_conc_mem' + suffix],
                          'x_out' + suffix)
    # storing infused feature
    # infused_feature = concept_selector * memory_feature
    # infused_feature = model.net.Mul([concept_selector, memory_feature],
    #                                'infused_feature' + suffix)

    #logits = model.FC('x_out' + suffix, 'logits' + suffix, cfg.OUTPUT_EMBEDDING_DIM, num_classes, weight_init=('GaussianFill', {'std': 0.01}), bias_init=('ConstantFill', {'value': 0.}))

    logits = add_cosnorm_classifier(model, 'x_out' + suffix, suffix, cfg.OUTPUT_EMBEDDING_DIM, num_classes)

    return logits  # , [direct_feature, infused_feature]


def add_hallucinator(model, input_blob_name, output_blob_name, feat_size, num_classes):
    out = model.FC(input_blob_name, output_blob_name,
                   feat_size, num_classes, weight_init=('GaussianFill', {'std': 0.01}),
                   bias_init=('ConstantFill', {'value': 0.}))
    return out


def add_selector(model, input_blob_name, output_blob_name, feat_size):
    out = model.FC(input_blob_name, output_blob_name,
                   feat_size, feat_size, weight_init=('GaussianFill', {'std': 0.01}),
                   bias_init=('ConstantFill', {'value': 0.}))
    return out


def add_cosnorm_classifier(model, input_, suffix, in_dims, out_dims):
    #  norm_x = torch.norm(input, 2, 1, keepdim=True)
    norm_x = l2_norm(model, input_, keepdims=True)
    model.net.Normalize(input_, 'input_normalized' + suffix)


    # ex = (norm_x / (1 + norm_x)) * (input / norm_x)
    model.net.Add([norm_x, 'one_blob'],
                  'one_plus_norm' + suffix, broadcast=1)  # (1 + norm_x)

    model.net.Div([norm_x, 'one_plus_norm' + suffix],
                  'norm_over_one_plus_norm' + suffix)  # (norm_x / (1 + norm_x))


    model.net.Mul(['input_normalized' + suffix, 'norm_over_one_plus_norm' + suffix],
                  'ex' + suffix, broadcast=1)  # (norm_x / (1 + norm_x)) * (input / norm_x)

    # ew = self.weight / torch.norm(self.weight, 2, 1, keepdim=True)
    model.net.Normalize('weight' + suffix, 'ew' + suffix)
    model.net.Mul(['ex' + suffix, 'scale_blob'],
                  'scaled_ex' + suffix, broadcast=1)
    out = model.net.MatMul(['scaled_ex' + suffix, 'ew' + suffix],
                           'logits' + suffix, trans_b=1)
    return out


def l2_norm(model, input_, keepdims=False):
    lp_vec_raised = model.net.Pow(input_, exponent=2.)
    lp_vec_summed = model.net.ReduceBackSum([lp_vec_raised], num_reduce_dims=1)
    lp_norm = model.net.Pow(lp_vec_summed, exponent=(1/2))
    if keepdims:
        lp_norm = model.net.ExpandDims(lp_norm, dims=[1])
    return lp_norm

def load_pickle(pickle_file):
    try:
        with open(pickle_file, 'rb') as f:
            pickle_data = pickle.load(f)
    except UnicodeDecodeError as e:
        with open(pickle_file, 'rb') as f:
            pickle_data = pickle.load(f, encoding='latin1')
    except Exception as e:
        print('Unable to load data ', pickle_file, ':', e)
        raise
    return pickle_data
