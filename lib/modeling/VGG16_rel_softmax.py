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

    x_sbj, x_obj, x_rel = add_visual_embedding(model, blob_sbj, dim_sbj, blob_obj, dim_obj,
                                                blob_rel_prd, dim_rel_prd,
                                                blob_rel_sbj, dim_rel_sbj,
                                                blob_rel_obj, dim_rel_obj)

    # load_centroids()
    model.net.ConstantFill([], 'one_blob', shape=[1], value=1.0)
    model.net.ConstantFill([], 'scale_blob', shape=[1], value=16.0)
    model.net.ConstantFill([], 'scale_10_blob', shape=[1], value=10.0)

    add_memory_module(model, x_sbj, 'centroids_obj', 'sbj', cfg.MODEL.NUM_CLASSES_SBJ_OBJ)
    add_memory_module(model, x_obj, 'centroids_obj', 'obj', cfg.MODEL.NUM_CLASSES_SBJ_OBJ)
    add_memory_module(model, x_rel, 'centroids_rel', 'rel', cfg.MODEL.NUM_CLASSES_PRD)
    # During testing, get topk labels and scores
    if not model.train:
        add_labels_and_scores_topk(model, 'sbj')
        add_labels_and_scores_topk(model, 'obj')
        add_labels_and_scores_topk(model, 'rel')

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
    x_sbj = model.add_FC_layer_with_weight_name(
            'x_sbj_and_obj',
            blob_sbj, 'x_sbj', dim_sbj, cfg.OUTPUT_EMBEDDING_DIM)
            # blob_sbj, 'x_sbj', dim_sbj, cfg.MODEL.NUM_CLASSES_SBJ_OBJ)
    x_obj = model.add_FC_layer_with_weight_name(
            'x_sbj_and_obj',
            blob_obj, 'x_obj', dim_obj, cfg.OUTPUT_EMBEDDING_DIM)
            # blob_obj, 'x_obj', dim_obj, cfg.MODEL.NUM_CLASSES_SBJ_OBJ)
    x_rel = model.FC(
            blob_rel_prd, 'x_rel',
            dim_rel_prd, cfg.OUTPUT_EMBEDDING_DIM)
            # dim_rel_prd, cfg.MODEL.NUM_CLASSES_PRD)

    # model.net.Alias('x_rel_prd_raw_1', 'x_rel_prd_raw')
    # model.net.Normalize('x_sbj_raw', 'x_sbj')
    # model.net.Normalize('x_obj_raw', 'x_obj')
    # model.net.Normalize('x_rel_prd_raw', 'x_rel')
    return x_sbj, x_obj, x_rel


def add_embd_pos_neg_splits(model, label):
    preprefix = label + '_'
    suffix = '_' + label

    if cfg.MODEL.SUBTYPE.find('xp_only') < 0:
        model.net.Slice(['logits' + suffix, preprefix + 'pos_starts',
                         preprefix + 'pos_ends'], 'xp' + suffix)
        model.Scale('xp' + suffix, 'scaled_xp' + suffix, scale=cfg.TRAIN.NORM_SCALAR)
    else:
        model.net.Alias('logits' + suffix, 'xp' + suffix)


def add_softmax_losses(model, label):
    prefix = label + '_'
    suffix = '_' + label

    _, loss_xp_yall = model.net.SoftmaxWithLoss(
        ['scaled_xp' + suffix, prefix + 'pos_labels_int32'],
        ['xp_yall_prob' + suffix, 'loss_xp_yall' + suffix],
        scale=1. / cfg.NUM_DEVICES)

    model.loss_set.extend([loss_xp_yall])


def add_labels_and_scores_topk(model, label):
    suffix = '_' + label
    model.net.TopK('logits' + suffix, ['scores' + suffix, 'labels' + suffix], k=250)


def add_memory_module(model, x, centroids_blob_name, label, num_classes):
    prefix = label + '_'
    suffix = '_' + label

    # storing direct feature
    direct_feature = x

    batch_size = 1
    feat_size = cfg.OUTPUT_EMBEDDING_DIM

    # set up visual memory
    # x_expand = x.unsqueeze(1).expand(-1, self.num_classes, -1)
    model.net.ExpandDims(['x' + suffix],
                         ['x_expanddims' + suffix],
                         dims=[1])

    model.net.Reshape(['x_expanddims' + suffix],
                      ['x_expand' + suffix, 'x_old_shape' + suffix],
                      shape=(-1, num_classes, -1))

    # centroids_expand = centroids.unsqueeze(0).expand(batch_size, -1, -1)
    model.net.ExpandDims([centroids_blob_name],
                         ['centroids_expanddims' + suffix],
                         dims=[0])

    model.net.Reshape([centroids_blob_name],
                      ['centroids_expand' + suffix, 'centroids_old_shape' + suffix],
                      shape=(batch_size, -1, -1))
    keys_memory = centroids_blob_name

    # computing reachability
    # dist_cur = torch.norm(x_expand - centroids_expand, 2, 2)
    model.net.Sub(['x_expand' + suffix, 'centroids_expand' + suffix],
                  'sub_x_expand_centroids_expand' + suffix)
    model.net.LpNorm(['sub_x_expand_centroids_expand' + suffix],
                     'dist_cur' + suffix, p=2)

    # values_nn, labels_nn = torch.sort(dist_cur, 1)
    # model.net.Sort(['dist_cur' + suffix],
    #                ['values_nn' + suffix, 'labels_nn' + suffix])  # TODO: not an actual function, to be done
    model.net.Min(['dist_cur' + suffix],
                  'min_dis' + suffix)

    # scale = 10.0
    # reachability = (scale / values_nn[:, 0]).unsqueeze(1).expand(-1, feat_size)

    # sliced_values_nn = model.net.Slice([values_nn], 'sliced_values_nn' + suffix, starts=[0, 0], ends=[-1, 1]) # TODO check if model.net.Slice() is the correct way to slice in caffe2
    model.net.Div(['scale_10_blob', 'min_dis' + suffix], 'scale_over_values'+ suffix, broadcast=1)
    model.net.ExpandDims(['scale_over_values' + suffix],
                         'scale_over_values_expand' + suffix,
                         dims=[1])

    reachability = model.net.Reshape(['scale_over_values_expand' + suffix],
                                      'reachability' + suffix,
                                      shape=(-1, feat_size))
    # computing memory feature by querying and associating visual memory

    # values_memory = self.fc_hallucinator(x)
    values_memory = add_hallucinator(model, 'x' + suffix, 'values_memory' + suffix, feat_size, num_classes)
    # values_memory = values_memory.softmax(dim=1)
    values_memory = model.net.softmax(values_memory, axis=1)
    # memory_feature = torch.matmul(values_memory, keys_memory)
    memory_feature = model.net.MatMul([values_memory, keys_memory],
                                      'memory_feature' + suffix, trans_b=1)

    # computing concept selector
    # concept_selector = self.fc_selector(x)
    concept_selector = add_selector(model, 'x' + suffix, 'concept_selector' + suffix, feat_size)
    # concept_selector = concept_selector.tanh()
    concept_selector = model.net.tanh(concept_selector)
    # x = reachability * (direct_feature + concept_selector * memory_feature)
    model.net.Mul([concept_selector, memory_feature],
                     'matmul_concep_memory' + suffix)
    model.net.Add([direct_feature, 'matmul_concep_memory' + suffix], 'add_matmul_conc_mem' + suffix)
    x_out = model.net.Mul([reachability, 'add_matmul_conc_mem' + suffix],
                             'x_out' + suffix)

    # storing infused feature
    # infused_feature = concept_selector * memory_feature
    infused_feature = model.net.Mul([concept_selector, memory_feature],
                                        'infused_feature' + suffix)

    logits = add_cosnorm_classifier(x_out, suffix, cfg.OUTPUT_EMBEDDING_DIM, num_classes)

    return logits, [direct_feature, infused_feature]


def add_hallucinator(model, input_blob_name, output_blob_name, feat_size, num_classes):
    out = model.FC(input_blob_name, output_blob_name,
                   feat_size, num_classes)
    return out


def add_selector(model, input_blob_name, output_blob_name, feat_size):
    out = model.FC(input_blob_name, output_blob_name,
                   feat_size, feat_size)
    return out


def add_cosnorm_classifier(input, suffix, in_dims, out_dims):
    model.net.LpNorm([input],
                     'norm' + suffix, p=2)

    # ex = (norm_x / (1 + norm_x)) * (input / norm_x)
    model.net.Add(['norm' + suffix, 'one_blob'],
                  'one_plus_norm' + suffix, broadcast=1)        # (1 + norm_x)

    model.net.Div(['norm' + suffix, 'one_plus_norm' + suffix],
                  'norm_over_one_plus_norm' + suffix)           # (norm_x / (1 + norm_x))

    model.net.Div([input, 'norm' + suffix],
                  'input_over_norm' + suffix)                   # (input / norm_x)

    model.net.Mul(['norm_over_one_plus_norm' + suffix,
                           'input_over_norm' + suffix],
                          'ex' + suffix)                        # (norm_x / (1 + norm_x)) * (input / norm_x)

    # ew = self.weight / torch.norm(self.weight, 2, 1, keepdim=True)
    model.net.LpNorm(['weight' + suffix], 'weight_norm' + suffix, p=2)
    model.net.Div(['weight' + suffix, 'weight_norm' + suffix],
                  'ew' + suffix)
    model.net.Mul(['scale_blob', 'ex' + suffix],
                     'scaled_ex' + suffix, broadcast=1)
    out = model.net.MatMul(['scaled_ex' + suffix, 'ew' + suffix],
                           'logits' + suffix, trans_b=1)
    return out


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

