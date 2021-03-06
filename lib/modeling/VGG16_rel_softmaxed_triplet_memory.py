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
import math
from modeling.VGG16_rel_softmax_memory import add_memory_module, add_hallucinator, add_selector, add_cosnorm_classifier, \
   disc_centroids_loss_func, add_centroids_loss, l2_norm

logger = logging.getLogger(__name__)

dropout_ratio = cfg.TRAIN.DROPOUT


def create_model(model):
    logger.info(' | VGG-16 yall softmaxed triplet {}'.format(cfg.DATASET))

    model.loss_set = []

    # 1. visual modules
    blob, dim, spatial_scale = add_VGG16_conv5_body(model)

    # stop gradient at conv 5:
    model.StopGradient(blob, blob)


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
        # model.StopGradient('centroids_obj', 'centroids_obj')
        # model.StopGradient('centroids_sbj', 'centroids_sbj')

        std = 1. / math.sqrt(cfg.OUTPUT_EMBEDDING_DIM)
        model.add_weight_blob_with_weight_name('weight_obj', cfg.MODEL.NUM_CLASSES_SBJ_OBJ, cfg.OUTPUT_EMBEDDING_DIM, -std, std)
        model.net.Alias('weight_obj', 'weight_sbj')

    if cfg.MODEL.MEMORY_MODULE_PRD:
        model.add_centroids_blob_with_weight_name('centroids_rel', cfg.MODEL.NUM_CLASSES_PRD, cfg.OUTPUT_EMBEDDING_DIM)
        #if cfg.DEBUG:
        #    if cfg.MODEL.SUBTYPE.find('xp_only') < 0:
        #        model.Scale('centroids_rel', 'centroids_rel', scale=cfg.TRAIN.NORM_SCALAR)
        # model.StopGradient('centroids_rel', 'centroids_rel')

        std = 1. / math.sqrt(cfg.OUTPUT_EMBEDDING_DIM)
        model.add_weight_blob_with_weight_name('weight_rel', cfg.MODEL.NUM_CLASSES_PRD, cfg.OUTPUT_EMBEDDING_DIM, -std, std)

    add_visual_embedding(
        model, blob_sbj, dim_sbj, blob_obj, dim_obj,
        blob_rel_prd, dim_rel_prd,
        blob_rel_sbj, dim_rel_sbj,
        blob_rel_obj, dim_rel_obj)

    # if cfg.MODEL.MEMORY_MODULE_SBJ_OBJ:
    x_blob_sbj_direct = 'x_sbj'
    x_blob_obj_direct = 'x_obj'

    x_blob_rel_sbj_direct = 'x_rel_sbj'
    x_blob_rel_obj_direct = 'x_rel_obj'
    # x_blob_rel = 'x_rel'

    # if cfg.MODEL.MEMORY_MODULE_SBJ_OBJ:
    #     model.StopGradient(x_blob_sbj, x_blob_sbj)
    #     model.StopGradient(x_blob_obj, x_blob_obj)
    # # if cfg.MODEL.MEMORY_MODULE_PRD:
    #     model.StopGradient(x_blob_rel, x_blob_rel)

    # Modulated Attention Proxy

    if cfg.MODEL.MEMORY_MODULE_SBJ_OBJ and cfg.MODEL.ATTENTION:
        x_blob_sbj_direct = model.add_FC_layer_with_weight_name('attention_sbj_obj', x_blob_sbj_direct, 'x_sbj_att',
                                                         cfg.OUTPUT_EMBEDDING_DIM, cfg.OUTPUT_EMBEDDING_DIM)
        x_blob_obj_direct = model.add_FC_layer_with_weight_name('attention_sbj_obj', x_blob_obj_direct, 'x_obj_att',
                                                         cfg.OUTPUT_EMBEDDING_DIM, cfg.OUTPUT_EMBEDDING_DIM)
        if model.train:
            x_blob_rel_sbj_direct = model.add_FC_layer_with_weight_name('attention_sbj_obj', x_blob_rel_sbj_direct, 'x_rel_sbj_att',
                                                             cfg.OUTPUT_EMBEDDING_DIM, cfg.OUTPUT_EMBEDDING_DIM)
            x_blob_rel_obj_direct = model.add_FC_layer_with_weight_name('attention_sbj_obj', x_blob_rel_obj_direct, 'x_rel_obj_att',
                                                             cfg.OUTPUT_EMBEDDING_DIM, cfg.OUTPUT_EMBEDDING_DIM)

    model.net.ConstantFill([], 'neg_two_blob', shape=[1], value=-2.0) # used for memory module

    if cfg.MODEL.MEMORY_MODULE_SBJ_OBJ:
        x_blob_sbj_mem = add_memory_module(model, x_blob_sbj_direct, 'centroids_sbj', 'sbj', cfg.MODEL.NUM_CLASSES_SBJ_OBJ)
        x_blob_obj_mem = add_memory_module(model, x_blob_obj_direct, 'centroids_obj', 'obj', cfg.MODEL.NUM_CLASSES_SBJ_OBJ)

        if model.train:
            x_blob_rel_sbj_mem = add_memory_module(model, x_blob_rel_sbj_direct, 'centroids_sbj', 'sbj', cfg.MODEL.NUM_CLASSES_SBJ_OBJ)
            x_blob_rel_obj_mem = add_memory_module(model, x_blob_rel_obj_direct, 'centroids_obj', 'obj', cfg.MODEL.NUM_CLASSES_SBJ_OBJ)

    # if cfg.MODEL.MEMORY_MODULE_PRD:
    #     v_meta_rel = add_memory_module(model, x_blob_rel, 'centroids_rel', 'rel', cfg.MODEL.NUM_CLASSES_PRD, batch_size_rel, scale_10_blob_rel)
    x_blob_rel_sbj = None
    x_blob_rel_obj = None

    if cfg.MODEL.MEMORY_MODULE_SBJ_OBJ:
        x_blob_sbj = x_blob_sbj_mem
        x_blob_obj = x_blob_obj_mem
        if model.train:
            x_blob_rel_sbj = x_blob_rel_sbj_mem
            x_blob_rel_obj = x_blob_rel_obj_mem
    else:
        x_blob_sbj = x_blob_sbj_direct
        x_blob_obj = x_blob_obj_direct
        if model.train:
            x_blob_rel_sbj = x_blob_rel_sbj_direct
            x_blob_rel_obj = x_blob_rel_obj_direct

    add_embd_fusion_for_p(model, x_blob_sbj, x_blob_obj, x_blob_rel_sbj, x_blob_rel_obj)
    x_blob_rel = 'x_rel'

    if cfg.MODEL.MEMORY_MODULE_SBJ_OBJ:
        x_blob_sbj = model.net.Normalize(x_blob_sbj)
        x_blob_obj = model.net.Normalize(x_blob_obj)
        x_blob_rel = model.net.Normalize(x_blob_rel)

    model.net.ConstantFill([], 'one_blob', shape=[1], value=1.0)
    model.net.ConstantFill([], 'scale_blob', shape=[1], value=16.0)

    add_language_embedding_for_vocab(model)

    # During testing, get topk labels and scores
    if not model.train:
        add_labels_and_scores_topk(model, 'sbj', x_blob_sbj)
        add_labels_and_scores_topk(model, 'obj', x_blob_obj)
        add_labels_and_scores_topk(model, 'rel', x_blob_rel)

    # # 2. language modules and losses
    if model.train:
        add_language_embedding_for_gt(model)

        add_embd_pos_neg_splits(model, 'sbj', x_blob_sbj)
        add_embd_pos_neg_splits(model, 'obj', x_blob_obj)
        add_embd_pos_neg_splits(model, 'rel', x_blob_rel)

        # define several helper blobs
        sbj_margin = cfg.TRAIN.MARGIN_SO
        obj_margin = cfg.TRAIN.MARGIN_SO
        rel_margin = cfg.TRAIN.MARGIN_P
        model.net.ConstantFill([], 'margin_blob_sbj', shape=[1], value=sbj_margin)
        model.net.ConstantFill([], 'margin_blob_obj', shape=[1], value=obj_margin)
        model.net.ConstantFill([], 'margin_blob_rel', shape=[1], value=rel_margin)
        model.net.ConstantFill([], 'one_blob', shape=[1], value=1.0)

        model.net.ConstantFill([], 'hubness_blob_sbj', shape=[1], value=1.0 / cfg.MODEL.NUM_CLASSES_SBJ_OBJ)
        model.net.ConstantFill([], 'hubness_blob_obj', shape=[1], value=1.0 / cfg.MODEL.NUM_CLASSES_SBJ_OBJ)
        model.net.ConstantFill([], 'hubness_blob_rel', shape=[1], value=1.0 / cfg.MODEL.NUM_CLASSES_PRD)

        add_embd_triplet_losses_labeled(model, 'sbj')
        add_embd_triplet_losses_labeled(model, 'obj')
        add_embd_triplet_losses_labeled(model, 'rel')

        if cfg.MODEL.MEMORY_MODULE_SBJ_OBJ:
            x_blob_sbj_direct = model.net.Slice([x_blob_sbj_direct, 'sbj_pos_starts', 'sbj_pos_ends'])
            #x_blob_sbj_direct = model.Scale(x_blob_sbj_direct, 'scaled_direct_sbj', scale=cfg.TRAIN.NORM_SCALAR)
            x_blob_obj_direct = model.net.Slice([x_blob_obj_direct, 'obj_pos_starts', 'obj_pos_ends'])
            #x_blob_obj_direct = model.Scale(x_blob_obj_direct, 'scaled_direct_obj', scale=cfg.TRAIN.NORM_SCALAR)

            add_centroids_loss(model, x_blob_sbj_direct, 'sbj', cfg.MODEL.NUM_CLASSES_SBJ_OBJ)
            add_centroids_loss(model, x_blob_obj_direct, 'obj', cfg.MODEL.NUM_CLASSES_SBJ_OBJ)

        if cfg.MODEL.MEMORY_MODULE_PRD:
            add_centroids_loss(model, x_blob_rel_direct, 'rel', cfg.MODEL.NUM_CLASSES_PRD)

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


def add_embd_fusion_for_p(model, x_blob_sbj, x_blob_obj, x_blob_rel_sbj, x_blob_rel_obj):
    if cfg.MODEL.SUBTYPE.find('embd_fusion') < 0:
        model.net.Alias('x_rel_prd', 'x_rel_raw_final')
    else:
        if model.train:
            x_spo = model.Concat(
                [x_blob_rel_sbj, x_blob_rel_obj, 'x_rel_prd'], 'x_spo')
            dim_x_spo = cfg.OUTPUT_EMBEDDING_DIM * 3
        else:
            x_spo = model.Concat(
                [x_blob_sbj, x_blob_obj, 'x_rel_prd'], 'x_spo')
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


def add_language_embedding_for_gt(model):
    sbj_vecs_name = 'sbj_pos_vecs'
    obj_vecs_name = 'obj_pos_vecs'
    rel_vecs_name = 'rel_pos_vecs'
    sbj_embd_name = 'sbj_pos_lan_embds'
    obj_embd_name = 'obj_pos_lan_embds'
    rel_embd_name = 'rel_pos_lan_embds'

    if cfg.TEXT_EMBEDDING.HIDDEN_LAYERS > 0:
        model.add_FC_layer_with_weight_name(
            'lang_sbj_and_obj',
            sbj_vecs_name,
            sbj_embd_name + '_raw_1',
            cfg.INPUT_LANG_EMBEDDING_DIM, cfg.OUTPUT_EMBEDDING_DIM,
            weight_init=('GaussianFill', {'std': 0.01}),
            bias_init=('ConstantFill', {'value': 0.}))
        model.add_FC_layer_with_weight_name(
            'lang_sbj_and_obj',
            obj_vecs_name,
            obj_embd_name + '_raw_1',
            cfg.INPUT_LANG_EMBEDDING_DIM, cfg.OUTPUT_EMBEDDING_DIM,
            weight_init=('GaussianFill', {'std': 0.01}),
            bias_init=('ConstantFill', {'value': 0.}))
        model.add_FC_layer_with_weight_name(
            'lang_rel',
            rel_vecs_name,
            rel_embd_name + '_raw_1',
            cfg.INPUT_LANG_EMBEDDING_DIM, cfg.OUTPUT_EMBEDDING_DIM,
            weight_init=('GaussianFill', {'std': 0.01}),
            bias_init=('ConstantFill', {'value': 0.}))
        if cfg.TEXT_EMBEDDING.HIDDEN_LAYERS > 1:
            model.Relu(sbj_embd_name + '_raw_1', sbj_embd_name + '_raw_1')
            model.add_FC_layer_with_weight_name(
                'lang_sbj_and_obj_2',
                sbj_embd_name + '_raw_1',
                sbj_embd_name + '_raw_2',
                cfg.OUTPUT_EMBEDDING_DIM, cfg.OUTPUT_EMBEDDING_DIM,
                weight_init=('GaussianFill', {'std': 0.01}),
                bias_init=('ConstantFill', {'value': 0.}))
            model.Relu(obj_embd_name + '_raw_1', obj_embd_name + '_raw_1')
            model.add_FC_layer_with_weight_name(
                'lang_sbj_and_obj_2',
                obj_embd_name + '_raw_1',
                obj_embd_name + '_raw_2',
                cfg.OUTPUT_EMBEDDING_DIM, cfg.OUTPUT_EMBEDDING_DIM,
                weight_init=('GaussianFill', {'std': 0.01}),
                bias_init=('ConstantFill', {'value': 0.}))
            model.Relu(rel_embd_name + '_raw_1', rel_embd_name + '_raw_1')
            model.add_FC_layer_with_weight_name(
                'lang_rel_2',
                rel_embd_name + '_raw_1',
                rel_embd_name + '_raw_2',
                cfg.OUTPUT_EMBEDDING_DIM, cfg.OUTPUT_EMBEDDING_DIM,
                weight_init=('GaussianFill', {'std': 0.01}),
                bias_init=('ConstantFill', {'value': 0.}))
            if cfg.TEXT_EMBEDDING.HIDDEN_LAYERS > 2:
                model.Relu(sbj_embd_name + '_raw_2', sbj_embd_name + '_raw_2')
                model.add_FC_layer_with_weight_name(
                    'lang_sbj_and_obj_3',
                    sbj_embd_name + '_raw_2',
                    sbj_embd_name + '_raw_3',
                    cfg.OUTPUT_EMBEDDING_DIM, cfg.OUTPUT_EMBEDDING_DIM,
                    weight_init=('GaussianFill', {'std': 0.01}),
                    bias_init=('ConstantFill', {'value': 0.}))
                model.Relu(obj_embd_name + '_raw_2', obj_embd_name + '_raw_2')
                model.add_FC_layer_with_weight_name(
                    'lang_sbj_and_obj_3',
                    obj_embd_name + '_raw_2',
                    obj_embd_name + '_raw_3',
                    cfg.OUTPUT_EMBEDDING_DIM, cfg.OUTPUT_EMBEDDING_DIM,
                    weight_init=('GaussianFill', {'std': 0.01}),
                    bias_init=('ConstantFill', {'value': 0.}))
                model.Relu(rel_embd_name + '_raw_2', rel_embd_name + '_raw_2')
                model.add_FC_layer_with_weight_name(
                    'lang_rel_3',
                    rel_embd_name + '_raw_2',
                    rel_embd_name + '_raw_3',
                    cfg.OUTPUT_EMBEDDING_DIM, cfg.OUTPUT_EMBEDDING_DIM,
                    weight_init=('GaussianFill', {'std': 0.01}),
                    bias_init=('ConstantFill', {'value': 0.}))
                # language embeddings must be normalized
                model.net.Normalize(sbj_embd_name + '_raw_3', sbj_embd_name)
                model.net.Normalize(obj_embd_name + '_raw_3', obj_embd_name)
                model.net.Normalize(rel_embd_name + '_raw_3', rel_embd_name)
            else:
                # language embeddings must be normalized
                model.net.Normalize(sbj_embd_name + '_raw_2', sbj_embd_name)
                model.net.Normalize(obj_embd_name + '_raw_2', obj_embd_name)
                model.net.Normalize(rel_embd_name + '_raw_2', rel_embd_name)
        else:
            # language embeddings must be normalized
            model.net.Normalize(sbj_embd_name + '_raw_1', sbj_embd_name)
            model.net.Normalize(obj_embd_name + '_raw_1', obj_embd_name)
            model.net.Normalize(rel_embd_name + '_raw_1', rel_embd_name)
    else:
        # everything is already normalized
        model.net.Alias(sbj_vecs_name, sbj_embd_name)
        model.net.Alias(obj_vecs_name, obj_embd_name)
        model.net.Alias(rel_vecs_name, rel_embd_name)


def add_language_embedding_for_vocab(model):
    if cfg.TEXT_EMBEDDING.HIDDEN_LAYERS > 0:
        model.add_FC_layer_with_weight_name(
            'lang_sbj_and_obj',
            'all_obj_word_vecs',
            'all_obj_lan_embds_raw_1',
            cfg.INPUT_LANG_EMBEDDING_DIM, cfg.OUTPUT_EMBEDDING_DIM,
            weight_init=('GaussianFill', {'std': 0.01}),
            bias_init=('ConstantFill', {'value': 0.}))
        model.add_FC_layer_with_weight_name(
            'lang_rel',
            'all_prd_word_vecs',
            'all_prd_lan_embds_raw_1',
            cfg.INPUT_LANG_EMBEDDING_DIM, cfg.OUTPUT_EMBEDDING_DIM,
            weight_init=('GaussianFill', {'std': 0.01}),
            bias_init=('ConstantFill', {'value': 0.}))
        if cfg.TEXT_EMBEDDING.HIDDEN_LAYERS > 1:
            model.Relu('all_obj_lan_embds_raw_1', 'all_obj_lan_embds_raw_1')
            model.add_FC_layer_with_weight_name(
                'lang_sbj_and_obj_2',
                'all_obj_lan_embds_raw_1',
                'all_obj_lan_embds_raw_2',
                cfg.OUTPUT_EMBEDDING_DIM, cfg.OUTPUT_EMBEDDING_DIM,
                weight_init=('GaussianFill', {'std': 0.01}),
                bias_init=('ConstantFill', {'value': 0.}))
            model.Relu('all_prd_lan_embds_raw_1', 'all_prd_lan_embds_raw_1')
            model.add_FC_layer_with_weight_name(
                'lang_rel_2',
                'all_prd_lan_embds_raw_1',
                'all_prd_lan_embds_raw_2',
                cfg.OUTPUT_EMBEDDING_DIM, cfg.OUTPUT_EMBEDDING_DIM,
                weight_init=('GaussianFill', {'std': 0.01}),
                bias_init=('ConstantFill', {'value': 0.}))
            if cfg.TEXT_EMBEDDING.HIDDEN_LAYERS > 2:
                model.Relu('all_obj_lan_embds_raw_2', 'all_obj_lan_embds_raw_2')
                model.add_FC_layer_with_weight_name(
                    'lang_sbj_and_obj_3',
                    'all_obj_lan_embds_raw_2',
                    'all_obj_lan_embds_raw_3',
                    cfg.OUTPUT_EMBEDDING_DIM, cfg.OUTPUT_EMBEDDING_DIM,
                    weight_init=('GaussianFill', {'std': 0.01}),
                    bias_init=('ConstantFill', {'value': 0.}))
                model.Relu('all_prd_lan_embds_raw_2', 'all_prd_lan_embds_raw_2')
                model.add_FC_layer_with_weight_name(
                    'lang_rel_3',
                    'all_prd_lan_embds_raw_2',
                    'all_prd_lan_embds_raw_3',
                    cfg.OUTPUT_EMBEDDING_DIM, cfg.OUTPUT_EMBEDDING_DIM,
                    weight_init=('GaussianFill', {'std': 0.01}),
                    bias_init=('ConstantFill', {'value': 0.}))
                # language embeddings must be normalized
                model.net.Normalize('all_obj_lan_embds_raw_3', 'all_obj_lan_embds')
                model.net.Normalize('all_prd_lan_embds_raw_3', 'all_prd_lan_embds')
            else:
                # language embeddings must be normalized
                model.net.Normalize('all_obj_lan_embds_raw_2', 'all_obj_lan_embds')
                model.net.Normalize('all_prd_lan_embds_raw_2', 'all_prd_lan_embds')
        else:
            model.net.Normalize('all_obj_lan_embds_raw_1', 'all_obj_lan_embds')
            model.net.Normalize('all_prd_lan_embds_raw_1', 'all_prd_lan_embds')
    else:
        model.net.Alias('all_obj_word_vecs', 'all_obj_lan_embds')
        model.net.Alias('all_prd_word_vecs', 'all_prd_lan_embds')

    model.Scale(
        'all_obj_lan_embds', 'scaled_all_obj_lan_embds', scale=cfg.TRAIN.NORM_SCALAR)
    model.Scale(
        'all_prd_lan_embds', 'scaled_all_prd_lan_embds', scale=cfg.TRAIN.NORM_SCALAR)


def add_embd_pos_neg_splits(model, label, x_blob, sublabel=''):
    preprefix = label + '_'
    if sublabel == '':
        prefix = preprefix
        suffix = '_' + label
    else:
        prefix = preprefix + sublabel + '_'
        suffix = '_' + label + '_' + sublabel

    if cfg.MODEL.SUBTYPE.find('xp_only') < 0:
        model.net.Slice([x_blob, preprefix + 'pos_starts',
                         preprefix + 'pos_ends'], 'xp' + suffix)
        model.Scale('xp' + suffix, 'scaled_xp' + suffix, scale=cfg.TRAIN.NORM_SCALAR)
        if suffix == '_rel':
            model.net.Slice(['x_rel_raw_final', prefix + 'pos_starts',
                             prefix + 'pos_ends'], 'xp_rel_raw_final')
        else:
            model.net.Slice(['x_' + label + '_raw', prefix + 'pos_starts',
                             prefix + 'pos_ends'], 'xp_' + label + '_raw')
    else:
        model.net.Alias(x_blob, 'xp' + suffix)
    model.net.Alias(prefix + 'pos_lan_embds', 'yp' + suffix)


def add_embd_triplet_losses_labeled(model, label):
    prefix = label + '_'
    suffix = '_' + label

    if label.find('rel') >= 0:
        num_neg_classes = len(model.roi_data_loader._landb._predicate_categories) - 1
    else:
        num_neg_classes = len(model.roi_data_loader._landb._object_categories) - 1
    model.net.ConstantFill(
        [], 'reciprocal_num_neg_classes_blob' + suffix,
        shape=[1], value=1.0 / float(num_neg_classes))

    model.net.SquaredL2Distance(['xp' + suffix, 'yp' + suffix],
                                'dist_xp_yp' + suffix)

    if label.find('rel') >= 0:
        yall_name = 'all_prd_lan_embds'
    else:
        yall_name = 'all_obj_lan_embds'

    model.net.MatMul(['scaled_xp' + suffix, 'scaled_' + yall_name],
                     'sim_xp_yall' + suffix, trans_b=1)

    if cfg.MODEL.WEAK_LABELS:
        if (label.find('rel') >= 0 and cfg.TRAIN.ADD_LOSS_WEIGHTS) or \
                ((label.find('sbj') >= 0 or label.find('obj') >= 0) and
                 cfg.TRAIN.ADD_LOSS_WEIGHTS_SO):
            _, loss_xp_yall = model.net.SoftmaxWithLoss(
                ['sim_xp_yall' + suffix,
                 prefix + 'pos_labels_float32_w',
                 prefix + 'pos_weights'],
                ['xp_yall_prob' + suffix, 'loss_xp_yall' + suffix],
                scale=1. / cfg.NUM_DEVICES,
                label_prob=1
            )

            model.loss_set.extend([loss_xp_yall])

        else:
            _, loss_xp_yall = model.net.SoftmaxWithLoss(
                ['sim_xp_yall' + suffix, prefix + 'pos_labels_float32_w'],
                ['xp_yall_prob' + suffix, 'loss_xp_yall' + suffix],
                scale=1. / cfg.NUM_DEVICES,
                label_prob=1
            )

            model.loss_set.extend([loss_xp_yall])
    else:
        if (label.find('rel') >= 0 and cfg.TRAIN.ADD_LOSS_WEIGHTS) or \
                ((label.find('sbj') >= 0 or label.find('obj') >= 0) and
                 cfg.TRAIN.ADD_LOSS_WEIGHTS_SO):
            _, loss_xp_yall = model.net.SoftmaxWithLoss(
                ['sim_xp_yall' + suffix,
                 prefix + 'pos_labels_int32',
                 prefix + 'pos_weights'],
                ['xp_yall_prob' + suffix, 'loss_xp_yall' + suffix],
                scale=1. / cfg.NUM_DEVICES)
        else:
            if cfg.MODEL.FOCAL_LOSS:
                if suffix in ['_sbj', '_obj']:
                    num_classes = cfg.MODEL.NUM_CLASSES_SBJ_OBJ
                if suffix in ['_rel']:
                    num_classes = cfg.MODEL.NUM_CLASSES_PRD
                model.net.Reshape(['sim_xp_yall' + suffix],
                                  ['sim_xp_yall_reshaped' + suffix, 'sim_xp_yall_old_shape' + suffix],
                                  shape=(0, 0, 1, 1))
                model.AddMetrics(['fg_num' + suffix, 'bg_num' + suffix])
                loss_xp_yall, _ = model.net.SoftmaxFocalLoss(
                    ['sim_xp_yall_reshaped' + suffix, prefix + 'pos_labels_int32',
                     'fg_num' + suffix],
                    ['loss_xp_yall' + suffix, 'xp_yall_prob' + suffix],
                    gamma=cfg.MODEL.FOCAL_LOSS_GAMMA,
                    alpha=cfg.MODEL.FOCAL_LOSS_ALPHA,
                    scale=1. / cfg.NUM_DEVICES,
                    num_classes=num_classes)

            else:
                _, loss_xp_yall = model.net.SoftmaxWithLoss(
                    ['sim_xp_yall' + suffix, prefix + 'pos_labels_int32'],
                    ['xp_yall_prob' + suffix, 'loss_xp_yall' + suffix],
                    scale=1. / cfg.NUM_DEVICES)

        if (cfg.TRAIN.HUBNESS):
            # pf = (1/batch_size) * sum(xp_yall_prob, axis=0)
            # xp_yall_prob  = Bp x K, where K is the number of classes, P is the positive image regions, B = (Bp+Bn) is the batch size, Bn is the negative image rengions
            model.Transpose(['xp_yall_prob' + suffix], ['xp_yall_probT' + suffix])
            # op = caffe2core.CreateOperator('Reshape_xpyall'+suffix, ['xp_yall_probT' + suffix], ['xp_yall_probT_reshaped'+suffix, 'old_shape'+suffix], shape=(0,1, -1) )
            # xp_yall_probT = K x Bp. This is Pij in https://www.aclweb.org/anthology/P19-1399
            model.net.Reshape(['xp_yall_probT' + suffix],
                              ['xp_yall_probT_reshaped' + suffix, 'xp_yall_probT_old_shape' + suffix],
                              shape=(0, 1, -1, 1))
            # xp_yall_probT_reshaped is K x 1 x Bp x 1
            xp_yall_probT_average_reshape_suffix = model.net.AveragePool(['xp_yall_probT_reshaped' + suffix], [
                'xp_yall_probT_average_reshape' + suffix], global_pooling=True)
            # op = core.CreateOperator('Reshape_xpyall_final'+suffix, ['xp_yall_probT_average_reshape' + suffix], ['xp_yall_probT_reshaped'+suffix, 'old_shape'+suffix], shape=(0,^) )
            # xp_yall_probT_average_reshape_suffix is K x 1 x 1 x 1
            # xp_yall_probT_average_reshape is pfj  in the paper https://www.aclweb.org/anthology/P19-1399
            hubness_dist_suffix = model.net.Sub(
                ['xp_yall_probT_average_reshape' + suffix, 'hubness_blob' + suffix], 'hubness_dist' + suffix,
                broadcast=1)
            hubness_dist_suffix_sqr = model.net.Sqr(['hubness_dist' + suffix], ['hubness_dist_sqr' + suffix])

            hubness_dist_suffix_sqr_scaled = model.Scale(['hubness_dist_sqr' + suffix],
                                                         ['hubness_dist_sqr_scaled' + suffix],
                                                         scale=cfg.TRAIN.HUBNESS_SCALE)

            # scale=scale
            # loss_hubness=   cfg.TRAIN.HUBNESS_SCALE* hubness_dist_suffix_sqr.AveragedLoss([], ['loss_hubness' + suffix])
            loss_hubness = hubness_dist_suffix_sqr_scaled.AveragedLoss([], ['loss_hubness' + suffix])
            # loss_hubness_scaled= loss_hubness.Scale(, scale=cfg.TRAIN.HUBNESS_SCALE, )
            model.loss_set.extend([loss_hubness])
            # loss_h = sum(pf - 1/k)

        model.loss_set.extend([loss_xp_yall])

    if cfg.MODEL.SPECS.find('no_xpypxn') < 0:
        model.net.MatMul(['yp' + suffix, 'x' + suffix],
                         'sim_yp_xall_raw' + suffix, trans_b=1)
        model.net.Sub(['sim_yp_xall_raw' + suffix, 'one_blob'],
                      'neg_dist_yp_xall' + suffix, broadcast=1)
        model.net.Add(['neg_dist_yp_xall' + suffix, 'dist_xp_yp' + suffix],
                      'diff_dist_xp_yp_xall' + suffix, broadcast=1, axis=0)
        model.net.Add(['diff_dist_xp_yp_xall' + suffix, 'margin_blob' + suffix],
                      'margin_xp_yp_xall' + suffix, broadcast=1)
        model.net.Relu('margin_xp_yp_xall' + suffix, 'max_margin_xp_yp_xall' + suffix)
        model.net.Mul(['max_margin_xp_yp_xall' + suffix, prefix + 'neg_affinity_mask'],
                      'max_margin_xp_yp_xn' + suffix)
        mean_max_margin_xp_yp_xn = model.net.ReduceBackMean(
            'max_margin_xp_yp_xn' + suffix, 'mean_max_margin_xp_yp_xn' + suffix)
        if (label.find('rel') >= 0 and cfg.TRAIN.ADD_LOSS_WEIGHTS) or \
                ((label.find('sbj') >= 0 or label.find('obj') >= 0) and
                 cfg.TRAIN.ADD_LOSS_WEIGHTS_SO):
            mean_max_margin_xp_yp_xn = model.net.Mul(
                ['mean_max_margin_xp_yp_xn' + suffix, prefix + 'pos_weights'],
                'mean_max_margin_xp_yp_xn_weighted' + suffix)
        loss_yp_xn = mean_max_margin_xp_yp_xn.AveragedLoss([], ['loss_yp_xn' + suffix])

        model.loss_set.extend([loss_yp_xn])

    if cfg.MODEL.SPECS.find('w_cluster') >= 0:
        # 1. get neg_dist_xp_xn
        model.net.MatMul(['xp' + suffix, 'x' + suffix],
                         'sim_xp_xall_raw' + suffix, trans_b=1)
        model.net.Sub(['sim_xp_xall_raw' + suffix, 'one_blob'],
                      'neg_dist_xp_xall' + suffix, broadcast=1)
        model.net.Mul(['neg_dist_xp_xall' + suffix, prefix + 'neg_affinity_mask'],
                      'neg_dist_xp_xn' + suffix)
        # 2. get mean_max_dist_xp_xp
        model.net.Negative('neg_dist_xp_xall' + suffix, 'dist_xp_xall' + suffix)
        model.net.Sub([prefix + 'neg_affinity_mask', 'one_blob'],
                      prefix + 'neg_pos_affinity_mask', broadcast=1)
        model.net.Negative(prefix + 'neg_pos_affinity_mask',
                           prefix + 'pos_affinity_mask')
        model.net.Mul(['dist_xp_xall' + suffix, prefix + 'pos_affinity_mask'],
                      'dist_xp_xp' + suffix)
        model.net.TopK('dist_xp_xp' + suffix,
                       ['max_dist_xp_xp' + suffix, '_idx_max_dist_xp_xp' + suffix], k=1)
        model.net.Squeeze('max_dist_xp_xp' + suffix,
                          'max_dist_xp_xp' + suffix, dims=[1])
        model.net.Add(['neg_dist_xp_xn' + suffix, 'max_dist_xp_xp' + suffix],
                      'diff_dist_xp_xp_xn' + suffix, broadcast=1, axis=0)
        model.net.Add(['diff_dist_xp_xp_xn' + suffix, 'margin_blob' + suffix],
                      'margin_xp_xp_xn' + suffix, broadcast=1)
        model.net.Relu('margin_xp_xp_xn' + suffix, 'max_margin_xp_xp_xn' + suffix)
        mean_max_margin_xp_xp_xn = model.net.ReduceBackMean(
            'max_margin_xp_xp_xn' + suffix, 'mean_max_margin_xp_xp_xn' + suffix)
        if (label.find('rel') >= 0 and cfg.TRAIN.ADD_LOSS_WEIGHTS) or \
                ((label.find('sbj') >= 0 or label.find('obj') >= 0) and
                 cfg.TRAIN.ADD_LOSS_WEIGHTS_SO):
            mean_max_margin_xp_xp_xn = model.net.Mul(
                ['mean_max_margin_xp_xp_xn' + suffix, prefix + 'pos_weights'],
                'mean_max_margin_xp_xp_xn_weighted' + suffix)
        loss_xp_xn = mean_max_margin_xp_xp_xn.AveragedLoss([], ['loss_xp_xn' + suffix])

        model.loss_set.extend([loss_xp_xn])


def add_labels_and_scores_topk(model, label, x_blob):
    suffix = '_' + label
    if label != 'rel':
        all_lan_embd = 'all_obj_lan_embds'
    else:
        all_lan_embd = 'all_prd_lan_embds'
    model.net.MatMul([x_blob, all_lan_embd], 'all_Y' + suffix, trans_b=1)
    model.net.TopK('all_Y' + suffix, ['scores' + suffix, 'labels' + suffix], k=250)
