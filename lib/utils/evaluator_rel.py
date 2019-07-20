# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# This file is based on 
# https://github.com/doubledaibo/drnet_cvpr2017/
# 
# Copyright the Chinese University of Hong Kong.
# See LICENSE file in the root directory of this source tree for details.

from __future__ import absolute_import
from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division

import numpy as np
import logging
import os
from collections import OrderedDict
import cPickle as pickle

from core.config_rel import cfg
from utils import helpers_rel
from utils import mypylib
from caffe2.python import workspace

import json

logger = logging.getLogger(__name__)

MIN_OVLP = 0.5


class Evaluator():

    def __init__(self, split, roidb_size):

        self._split = split

        self.spo_cnt = 0
        self.tri_top1_cnt = 0
        self.tri_top5_cnt = 0
        self.tri_top10_cnt = 0
        self.sbj_top1_cnt = 0
        self.sbj_top5_cnt = 0
        self.sbj_top10_cnt = 0
        self.obj_top1_cnt = 0
        self.obj_top5_cnt = 0
        self.obj_top10_cnt = 0
        self.rel_top1_cnt = 0
        self.rel_top5_cnt = 0
        self.rel_top10_cnt = 0

        if cfg.MODEL.WEAK_LABELS:
            self.tri_top1_w_cnt = 0
            self.sbj_top1_w_cnt = 0
            self.obj_top1_w_cnt = 0
            self.rel_top1_w_cnt = 0

            self.tri_top1_w_acc = 0.0
            self.sbj_top1_w_acc = 0.0
            self.obj_top1_w_acc = 0.0
            self.rel_top1_w_acc = 0.0

        self.tri_top1_acc = 0.0
        self.tri_top5_acc = 0.0
        self.tri_top10_acc = 0.0
        self.sbj_top1_acc = 0.0
        self.sbj_top5_acc = 0.0
        self.sbj_top10_acc = 0.0
        self.obj_top1_acc = 0.0
        self.obj_top5_acc = 0.0
        self.obj_top10_acc = 0.0
        self.rel_top1_acc = 0.0
        self.rel_top5_acc = 0.0
        self.rel_top10_acc = 0.0

        self.tri_rr = 0.0
        self.sbj_rr = 0.0
        self.obj_rr = 0.0
        self.rel_rr = 0.0

        self.tri_mr = 0.0
        self.sbj_mr = 0.0
        self.obj_mr = 0.0
        self.rel_mr = 0.0

        self.rank_k = 250
        self.all_rel_k = [1, 10, 70]

        self.det_list = \
            ['image_id', 'image_idx',
             'boxes_sbj', 'boxes_obj', 'boxes_rel',
             'labels_sbj', 'labels_obj', 'labels_rel',
             'scores_sbj', 'scores_obj', 'scores_rel',
             'gt_labels_sbj', 'gt_labels_obj', 'gt_labels_rel',
             'gt_boxes_sbj', 'gt_boxes_obj', 'gt_boxes_rel']

        if cfg.MODEL.WEAK_LABELS:
            self.det_list.append('gt_labels_sbj_w')
            self.det_list.append('gt_labels_obj_w')
            self.det_list.append('gt_labels_rel_w')

        self.all_dets = {key: [] for key in self.det_list}

        self.roidb_size = roidb_size
        self.tested = [set() for i in range(roidb_size)]

        self.all_det_labels = [[] for _ in self.all_rel_k]
        self.all_det_boxes = [[] for _ in self.all_rel_k]
        self.all_gt_labels = [[] for _ in self.all_rel_k]
        self.all_gt_boxes = [[] for _ in self.all_rel_k]

        if cfg.TEST.GET_ALL_LAN_EMBEDDINGS:
            self.all_obj_lan_embds = None
            self.all_prd_lan_embds = None
        if cfg.TEST.GET_ALL_VIS_EMBEDDINGS:
            self.all_sbj_vis_embds = []
            self.all_obj_vis_embds = []
            self.all_prd_vis_embds = []

        self._data_path = os.path.join(cfg.DATA_DIR, 'Visual_Genome')

        self._object_classes = []
        with open(self._data_path + '/object_categories_spo_joined_and_merged.txt') as obj_classes:
            for line in obj_classes:
                self._object_classes.append(line[:-1])
        self._num_object_classes = len(self._object_classes)
        self._object_class_to_ind = \
            dict(zip(self._object_classes, range(self._num_object_classes)))

        self._predicate_classes = []
        with open(self._data_path + '/predicate_categories_spo_joined_and_merged.txt') as prd_classes:
            for line in prd_classes:
                self._predicate_classes.append(line[:-1])
        self._num_predicate_classes = len(self._predicate_classes)
        self._predicate_class_to_ind = \
            dict(zip(self._predicate_classes, range(self._num_predicate_classes)))

        with open(self._data_path + '/words_synsets.json', 'r') as f:
            self._word_to_synset = json.load(f)



    def is_in(self, word_id, l_ids, word_type):
        if word_type == 'v':
            synset = self._word_to_synset['verbs'][self._predicate_classes[word_id]]
            l = [self._word_to_synset['verbs'][self._predicate_classes[l_id]] for l_id in l_ids]
        if word_type == 'o':
            synset = self._word_to_synset['nouns'][self._object_classes[word_id]]
            l = [self._word_to_synset['nouns'][self._object_classes[l_id]] for l_id in l_ids]

        for s in l:
            if mypylib.isA(synset, s)[0] == 1:
                return True
        else:
            return False

    def reset(self):
        # this should clear out all the metrics computed so far except the
        # best_topN metrics
        logger.info('Resetting {} evaluator...'.format(self._split))
        self.spo_cnt = 0
        self.tri_top1_cnt = 0
        self.tri_top5_cnt = 0
        self.tri_top10_cnt = 0
        self.sbj_top1_cnt = 0
        self.sbj_top5_cnt = 0
        self.sbj_top10_cnt = 0
        self.obj_top1_cnt = 0
        self.obj_top5_cnt = 0
        self.obj_top10_cnt = 0
        self.rel_top1_cnt = 0
        self.rel_top5_cnt = 0
        self.rel_top10_cnt = 0

        if cfg.MODEL.WEAK_LABELS:
            self.tri_top1_w_cnt = 0
            self.sbj_top1_w_cnt = 0
            self.obj_top1_w_cnt = 0
            self.rel_top1_w_cnt = 0

            self.tri_top1_w_acc = 0.0
            self.sbj_top1_w_acc = 0.0
            self.obj_top1_w_acc = 0.0
            self.rel_top1_w_acc = 0.0

        self.tri_top1_acc = 0.0
        self.tri_top5_acc = 0.0
        self.tri_top10_acc = 0.0
        self.sbj_top1_acc = 0.0
        self.sbj_top5_acc = 0.0
        self.sbj_top10_acc = 0.0
        self.obj_top1_acc = 0.0
        self.obj_top5_acc = 0.0
        self.obj_top10_acc = 0.0
        self.rel_top1_acc = 0.0
        self.rel_top5_acc = 0.0
        self.rel_top10_acc = 0.0

        self.tri_rr = 0.0
        self.sbj_rr = 0.0
        self.obj_rr = 0.0
        self.rel_rr = 0.0

        self.tri_mr = 0.0
        self.sbj_mr = 0.0
        self.obj_mr = 0.0
        self.rel_mr = 0.0

        self.all_dets = {key: [] for key in self.det_list}

        self.tested = [set() for i in range(self.roidb_size)]

        self.all_det_labels = [[] for _ in self.all_rel_k]
        self.all_det_boxes = [[] for _ in self.all_rel_k]
        self.all_gt_labels = [[] for _ in self.all_rel_k]
        self.all_gt_boxes = [[] for _ in self.all_rel_k]

        if cfg.TEST.GET_ALL_LAN_EMBEDDINGS:
            self.all_obj_lan_embds = None
            self.all_prd_lan_embds = None
        if cfg.TEST.GET_ALL_VIS_EMBEDDINGS:
            self.all_sbj_vis_embds = []
            self.all_obj_vis_embds = []
            self.all_prd_vis_embds = []

    def eval_im_dets_triplet_topk(self):

        prefix = 'gpu_' if cfg.DEVICE == 'GPU' else 'cpu_'

        if cfg.TEST.GET_ALL_LAN_EMBEDDINGS:
            if self.all_obj_lan_embds is None:
                self.all_obj_lan_embds = workspace.FetchBlob(
                    prefix + '{}/{}'.format(cfg.ROOT_DEVICE_ID, 'all_obj_lan_embds'))
            if self.all_prd_lan_embds is None:
                self.all_prd_lan_embds = workspace.FetchBlob(
                    prefix + '{}/{}'.format(cfg.ROOT_DEVICE_ID, 'all_prd_lan_embds'))

        new_batch_flag = False
        for gpu_id in range(cfg.ROOT_DEVICE_ID, cfg.ROOT_DEVICE_ID + cfg.NUM_DEVICES):

            image_idx = workspace.FetchBlob(
                prefix + '{}/{}'.format(gpu_id, 'image_idx'))[0]
            subbatch_id = workspace.FetchBlob(
                prefix + '{}/{}'.format(gpu_id, 'subbatch_id'))[0]
            if subbatch_id in self.tested[image_idx]:
                continue
            new_batch_flag = True
            self.tested[image_idx].add(subbatch_id)

            self.all_dets['image_idx'].append(int(image_idx))
            image_id = workspace.FetchBlob(
                prefix + '{}/{}'.format(gpu_id, 'image_id'))[0]
            self.all_dets['image_id'].append(image_id)

            scale = \
                workspace.FetchBlob(prefix + '{}/{}'.format(gpu_id, 'image_scale'))[0]
            gt_labels_sbj = workspace.FetchBlob(prefix + '{}/{}'.format(
                gpu_id, 'sbj_pos_labels_int32'))
            gt_labels_obj = workspace.FetchBlob(prefix + '{}/{}'.format(
                gpu_id, 'obj_pos_labels_int32'))
            gt_labels_rel = workspace.FetchBlob(prefix + '{}/{}'.format(
                gpu_id, 'rel_pos_labels_int32'))

            if cfg.MODEL.WEAK_LABELS:
                gt_labels_sbj_w = []
                gt_labels_obj_w = []
                gt_labels_rel_w = []
                for num_w in range(cfg.MODEL.NUM_WEAK_LABELS):
                    gt_labels_sbj_w.append(workspace.FetchBlob(prefix + '{}/{}'.format(
                        gpu_id, 'sbj_pos_labels_int32_w_' + str(num_w))))
                    gt_labels_obj_w.append(workspace.FetchBlob(prefix + '{}/{}'.format(
                        gpu_id, 'obj_pos_labels_int32_w_' + str(num_w))))
                    gt_labels_rel_w.append(workspace.FetchBlob(prefix + '{}/{}'.format(
                        gpu_id, 'rel_pos_labels_int32_w_' + str(num_w))))

                gt_labels_sbj_w = np.array(gt_labels_sbj_w)
                gt_labels_obj_w = np.array(gt_labels_obj_w)
                gt_labels_rel_w = np.array(gt_labels_rel_w)
                gt_labels_sbj_w -= 1
                gt_labels_obj_w -= 1
                gt_labels_rel_w -= 1

            gt_labels_sbj -= 1
            gt_labels_obj -= 1
            gt_labels_rel -= 1
            gt_boxes_sbj = workspace.FetchBlob(prefix + '{}/{}'.format(
                gpu_id, 'sbj_gt_boxes')) / scale
            gt_boxes_obj = workspace.FetchBlob(prefix + '{}/{}'.format(
                gpu_id, 'obj_gt_boxes')) / scale
            gt_boxes_rel = workspace.FetchBlob(prefix + '{}/{}'.format(
                gpu_id, 'rel_gt_boxes')) / scale
            self.all_dets['gt_labels_sbj'].append(gt_labels_sbj)
            self.all_dets['gt_labels_obj'].append(gt_labels_obj)
            self.all_dets['gt_labels_rel'].append(gt_labels_rel)

            if cfg.MODEL.WEAK_LABELS:
                self.all_dets['gt_labels_sbj_w'].append(gt_labels_sbj_w)
                self.all_dets['gt_labels_obj_w'].append(gt_labels_obj_w)
                self.all_dets['gt_labels_rel_w'].append(gt_labels_rel_w)

            self.all_dets['gt_boxes_sbj'].append(gt_boxes_sbj)
            self.all_dets['gt_boxes_obj'].append(gt_boxes_obj)
            self.all_dets['gt_boxes_rel'].append(gt_boxes_rel)

            num_proposals = int(workspace.FetchBlob(
                prefix + '{}/{}'.format(gpu_id, 'num_proposals'))[0])
            if num_proposals == 0:
                det_boxes_sbj = np.empty((0, 4), dtype=np.float32)
                det_boxes_obj = np.empty((0, 4), dtype=np.float32)
                det_boxes_rel = np.empty((0, 4), dtype=np.float32)
                det_labels_sbj = np.empty((0, 20), dtype=np.int32)
                det_labels_obj = np.empty((0, 20), dtype=np.int32)
                det_labels_rel = np.empty((0, 20), dtype=np.int32)
                det_scores_sbj = np.empty((0, 20), dtype=np.float32)
                det_scores_obj = np.empty((0, 20), dtype=np.float32)
                det_scores_rel = np.empty((0, 20), dtype=np.float32)
            else:
                det_boxes_sbj = workspace.FetchBlob(prefix + '{}/{}'.format(
                    gpu_id, 'sbj_rois'))[:, 1:] / scale
                det_boxes_obj = workspace.FetchBlob(prefix + '{}/{}'.format(
                    gpu_id, 'obj_rois'))[:, 1:] / scale
                det_boxes_rel = workspace.FetchBlob(prefix + '{}/{}'.format(
                    gpu_id, 'rel_rois_prd'))[:, 1:] / scale
                det_labels_sbj = \
                    workspace.FetchBlob(prefix + '{}/{}'.format(gpu_id, 'labels_sbj'))
                det_labels_obj = \
                    workspace.FetchBlob(prefix + '{}/{}'.format(gpu_id, 'labels_obj'))
                det_labels_rel = \
                    workspace.FetchBlob(prefix + '{}/{}'.format(gpu_id, 'labels_rel'))
                det_scores_sbj = \
                    workspace.FetchBlob(prefix + '{}/{}'.format(gpu_id, 'scores_sbj'))
                det_scores_obj = \
                    workspace.FetchBlob(prefix + '{}/{}'.format(gpu_id, 'scores_obj'))
                det_scores_rel = \
                    workspace.FetchBlob(prefix + '{}/{}'.format(gpu_id, 'scores_rel'))
            self.all_dets['boxes_sbj'].append(det_boxes_sbj)
            self.all_dets['boxes_obj'].append(det_boxes_obj)
            self.all_dets['boxes_rel'].append(det_boxes_rel)
            self.all_dets['labels_sbj'].append(det_labels_sbj)
            self.all_dets['labels_obj'].append(det_labels_obj)
            self.all_dets['labels_rel'].append(det_labels_rel)
            self.all_dets['scores_sbj'].append(det_scores_sbj)
            self.all_dets['scores_obj'].append(det_scores_obj)
            self.all_dets['scores_rel'].append(det_scores_rel)

            if cfg.TEST.GET_ALL_VIS_EMBEDDINGS:
                embds_sbj = workspace.FetchBlob(prefix + '{}/{}'.format(
                    gpu_id, 'x_sbj'))
                embds_obj = workspace.FetchBlob(prefix + '{}/{}'.format(
                    gpu_id, 'x_obj'))
                embds_prd = workspace.FetchBlob(prefix + '{}/{}'.format(
                    gpu_id, 'x_rel'))
                self.all_sbj_vis_embds.append(embds_sbj)
                self.all_obj_vis_embds.append(embds_obj)
                self.all_prd_vis_embds.append(embds_prd)

            if True:
                self.spo_cnt += len(gt_labels_sbj)
                for ind in range(len(gt_labels_sbj)):
                    if cfg.TEST.ISA:
                        if self.is_in(gt_labels_sbj[ind], det_labels_sbj[ind, :1], word_type='o') and \
                                self.is_in(gt_labels_obj[ind], det_labels_obj[ind, :1], word_type='o') and \
                                self.is_in(gt_labels_rel[ind], det_labels_rel[ind, :1], word_type='v'):
                            self.tri_top1_cnt += 1
                        if self.is_in(gt_labels_sbj[ind], det_labels_sbj[ind, :1], word_type='o'):
                            self.sbj_top1_cnt += 1
                        if self.is_in(gt_labels_obj[ind], det_labels_obj[ind, :1], word_type='o'):
                            self.obj_top1_cnt += 1
                        if self.is_in(gt_labels_rel[ind], det_labels_rel[ind, :1], word_type='v'):
                            self.rel_top1_cnt += 1

                        if self.is_in(gt_labels_sbj[ind], det_labels_sbj[ind, :5], word_type='o') and \
                                self.is_in(gt_labels_obj[ind], det_labels_obj[ind, :5], word_type='o') and \
                                self.is_in(gt_labels_rel[ind], det_labels_rel[ind, :5], word_type='v'):
                            self.tri_top5_cnt += 1
                        if self.is_in(gt_labels_sbj[ind], det_labels_sbj[ind, :5], word_type='o'):
                            self.sbj_top5_cnt += 1
                        if self.is_in(gt_labels_obj[ind], det_labels_obj[ind, :5], word_type='o'):
                            self.obj_top5_cnt += 1
                        if self.is_in(gt_labels_rel[ind], det_labels_rel[ind, :5], word_type='v'):
                            self.rel_top5_cnt += 1

                        if self.is_in(gt_labels_sbj[ind], det_labels_sbj[ind, :10], word_type='o') and \
                                self.is_in(gt_labels_obj[ind], det_labels_obj[ind, :10], word_type='o') and \
                                self.is_in(gt_labels_rel[ind], det_labels_rel[ind, :10], word_type='v'):
                            self.tri_top10_cnt += 1
                        if self.is_in(gt_labels_sbj[ind], det_labels_sbj[ind, :10], word_type='o'):
                            self.sbj_top10_cnt += 1
                        if self.is_in(gt_labels_obj[ind], det_labels_obj[ind, :10], word_type='o'):
                            self.obj_top10_cnt += 1
                        if self.is_in(gt_labels_rel[ind], det_labels_rel[ind, :10], word_type='v'):
                            self.rel_top10_cnt += 1

                        if cfg.MODEL.WEAK_LABELS:
                            if (self.is_in(det_labels_sbj[ind, 0:1], gt_labels_sbj_w[:, ind], word_type='o') or self.is_in(
                                    gt_labels_sbj[
                                        ind], det_labels_sbj[ind, :1], word_type='o')) and (
                                    self.is_in(det_labels_obj[ind, 0:1], gt_labels_obj_w[:, ind],
                                               word_type='o') or self.is_in(gt_labels_obj[
                                                                           ind], det_labels_obj[ind, :1],
                                                                       word_type='o')) and (
                                    self.is_in(det_labels_rel[ind, 0:1], gt_labels_rel_w[:, ind],
                                               word_type='v') or self.is_in(gt_labels_rel[
                                                                           ind], det_labels_rel[ind, :1], word_type='v')):
                                self.tri_top1_w_cnt += 1
                            if self.is_in(det_labels_sbj[ind, 0:1], gt_labels_sbj_w[:, ind], word_type='o') or self.is_in(
                                    gt_labels_sbj[
                                        ind], det_labels_sbj[ind, :1], word_type='o'):
                                self.sbj_top1_w_cnt += 1
                            if self.is_in(det_labels_obj[ind, 0:1], gt_labels_obj_w[:, ind]) or self.is_in(
                                    gt_labels_obj[
                                        ind], det_labels_obj[ind, :1], word_type='o'):
                                self.obj_top1_w_cnt += 1
                            if self.is_in(det_labels_rel[ind, 0:1], gt_labels_rel_w[:, ind]) or self.is_in(
                                    gt_labels_rel[
                                        ind], det_labels_rel[ind, :1], word_type='v'):
                                self.rel_top1_w_cnt += 1

                    else:
                        if gt_labels_sbj[ind] in det_labels_sbj[ind, :1] and \
                                gt_labels_obj[ind] in det_labels_obj[ind, :1] and \
                                gt_labels_rel[ind] in det_labels_rel[ind, :1]:
                            self.tri_top1_cnt += 1
                        if gt_labels_sbj[ind] in det_labels_sbj[ind, :1]:
                            self.sbj_top1_cnt += 1
                        if gt_labels_obj[ind] in det_labels_obj[ind, :1]:
                            self.obj_top1_cnt += 1
                        if gt_labels_rel[ind] in det_labels_rel[ind, :1]:
                            self.rel_top1_cnt += 1

                        if gt_labels_sbj[ind] in det_labels_sbj[ind, :5] and \
                                gt_labels_obj[ind] in det_labels_obj[ind, :5] and \
                                gt_labels_rel[ind] in det_labels_rel[ind, :5]:
                            self.tri_top5_cnt += 1
                        if gt_labels_sbj[ind] in det_labels_sbj[ind, :5]:
                            self.sbj_top5_cnt += 1
                        if gt_labels_obj[ind] in det_labels_obj[ind, :5]:
                            self.obj_top5_cnt += 1
                        if gt_labels_rel[ind] in det_labels_rel[ind, :5]:
                            self.rel_top5_cnt += 1

                        if gt_labels_sbj[ind] in det_labels_sbj[ind, :10] and \
                                gt_labels_obj[ind] in det_labels_obj[ind, :10] and \
                                gt_labels_rel[ind] in det_labels_rel[ind, :10]:
                            self.tri_top10_cnt += 1
                        if gt_labels_sbj[ind] in det_labels_sbj[ind, :10]:
                            self.sbj_top10_cnt += 1
                        if gt_labels_obj[ind] in det_labels_obj[ind, :10]:
                            self.obj_top10_cnt += 1
                        if gt_labels_rel[ind] in det_labels_rel[ind, :10]:
                            self.rel_top10_cnt += 1

                        if cfg.MODEL.WEAK_LABELS:
                            if (det_labels_sbj[ind, 0:1] in gt_labels_sbj_w[:, ind] or gt_labels_sbj[
                                ind] in det_labels_sbj[ind, :1]) and (
                                    det_labels_obj[ind, 0:1] in gt_labels_obj_w[:, ind] or gt_labels_obj[
                                ind] in det_labels_obj[ind, :1]) and (
                                    det_labels_rel[ind, 0:1] in gt_labels_rel_w[:, ind] or gt_labels_rel[
                                ind] in det_labels_rel[ind, :1]):
                                self.tri_top1_w_cnt += 1
                            if det_labels_sbj[ind, 0:1] in gt_labels_sbj_w[:, ind] or gt_labels_sbj[
                                ind] in det_labels_sbj[ind, :1]:
                                self.sbj_top1_w_cnt += 1
                            if det_labels_obj[ind, 0:1] in gt_labels_obj_w[:, ind] or gt_labels_obj[
                                ind] in det_labels_obj[ind, :1]:
                                self.obj_top1_w_cnt += 1
                            if det_labels_rel[ind, 0:1] in gt_labels_rel_w[:, ind] or gt_labels_rel[
                                ind] in det_labels_rel[ind, :1]:
                                self.rel_top1_w_cnt += 1

                    s_correct = gt_labels_sbj[ind] in det_labels_sbj[ind, :self.rank_k]
                    p_correct = gt_labels_rel[ind] in det_labels_rel[ind, :self.rank_k]
                    o_correct = gt_labels_obj[ind] in det_labels_obj[ind, :self.rank_k]
                    spo_correct = s_correct and p_correct and o_correct
                    s_ind = np.where(
                        det_labels_sbj[ind, :self.rank_k].squeeze() == \
                        gt_labels_sbj[ind])[0]
                    p_ind = np.where(
                        det_labels_rel[ind, :self.rank_k].squeeze() == \
                        gt_labels_rel[ind])[0]
                    o_ind = np.where(
                        det_labels_obj[ind, :self.rank_k].squeeze() == \
                        gt_labels_obj[ind])[0]

                    self.sbj_mr += 1
                    self.rel_mr += 1
                    self.obj_mr += 1
                    self.tri_mr += 1
                    if s_correct:
                        s_ind = s_ind[0]
                        self.sbj_rr += 1.0 / (s_ind + 1.0)
                        self.sbj_mr += s_ind / float(self.rank_k) - 1
                    if p_correct:
                        p_ind = p_ind[0]
                        self.rel_rr += 1.0 / (p_ind + 1.0)
                        self.rel_mr += p_ind / float(self.rank_k) - 1
                    if o_correct:
                        o_ind = o_ind[0]
                        self.obj_rr += 1.0 / (o_ind + 1.0)
                        self.obj_mr += o_ind / float(self.rank_k) - 1
                    if spo_correct:
                        self.tri_rr += (1.0 / (s_ind + 1.0) + \
                                        1.0 / (p_ind + 1.0) + \
                                        1.0 / (o_ind + 1.0)) / 3.0
                        self.tri_mr += (s_ind / float(self.rank_k) - 1 + \
                                        p_ind / float(self.rank_k) - 1 + \
                                        o_ind / float(self.rank_k) - 1) / 3.0

            sbj_k = 1
            # rel_k = 70
            obj_k = 1
            # det_labels = []
            # det_boxes = []
            # gt_labels = []
            # gt_boxes = []
            for i, rel_k in enumerate(self.all_rel_k):
                if det_labels_sbj.shape[0] > 0:
                    topk_labels_sbj = det_labels_sbj[:, :sbj_k]
                    topk_labels_rel = det_labels_rel[:, :rel_k]
                    topk_labels_obj = det_labels_obj[:, :obj_k]
                else:  # In the ECCV2016 proposals sometimes there is no det box
                    topk_labels_sbj = np.zeros((0, sbj_k), dtype=np.int32)
                    topk_labels_rel = np.zeros((0, rel_k), dtype=np.int32)
                    topk_labels_obj = np.zeros((0, obj_k), dtype=np.int32)

                if det_scores_sbj.shape[0] > 0:
                    topk_scores_sbj = det_scores_sbj[:, :sbj_k]
                    topk_scores_rel = det_scores_rel[:, :rel_k]
                    topk_scores_obj = det_scores_obj[:, :obj_k]
                else:  # In the ECCV2016 proposals sometimes there is no det box
                    topk_scores_sbj = np.zeros((0, sbj_k), dtype=np.float32)
                    topk_scores_rel = np.zeros((0, rel_k), dtype=np.float32)
                    topk_scores_obj = np.zeros((0, obj_k), dtype=np.float32)

                topk_cube_spo_labels = np.zeros(
                    (topk_labels_sbj.shape[0], sbj_k * obj_k * rel_k, 3), dtype=np.int32)
                topk_cube_spo_scores = np.zeros(
                    (topk_labels_sbj.shape[0], sbj_k * obj_k * rel_k), dtype=np.float32)
                topk_cube_p_scores = np.zeros(
                    (topk_labels_sbj.shape[0], sbj_k * obj_k * rel_k), dtype=np.float32)
                for l in range(sbj_k):
                    for m in range(rel_k):
                        for n in range(obj_k):
                            topk_cube_spo_labels[:, l * rel_k * obj_k + m * obj_k + n, 0] = \
                                topk_labels_sbj[:, l]
                            topk_cube_spo_labels[:, l * rel_k * obj_k + m * obj_k + n, 1] = \
                                topk_labels_rel[:, m]
                            topk_cube_spo_labels[:, l * rel_k * obj_k + m * obj_k + n, 2] = \
                                topk_labels_obj[:, n]
                            topk_cube_spo_scores[:, l * rel_k * obj_k + m * obj_k + n] = \
                                np.exp(topk_scores_sbj[:, l] +
                                       topk_scores_rel[:, m] +
                                       topk_scores_obj[:, n])

                            topk_cube_p_scores[:, l * rel_k * obj_k + m * obj_k + n] = \
                                np.exp(topk_scores_rel[:, m])

                topk_cube_spo_labels_reshape = topk_cube_spo_labels.reshape((-1, 3))
                topk_cube_spo_scores_reshape = topk_cube_spo_scores.reshape((-1, 1))

                self.all_det_labels[i].append(
                    np.concatenate((topk_cube_spo_scores_reshape[:, 0, np.newaxis],
                                    topk_cube_spo_labels_reshape[:, 0, np.newaxis],
                                    topk_cube_spo_labels_reshape[:, 1, np.newaxis],
                                    topk_cube_spo_labels_reshape[:, 2, np.newaxis]),
                                   axis=1))
                self.all_det_boxes[i].append(np.repeat(
                    np.concatenate((det_boxes_sbj[:, np.newaxis, :],
                                    det_boxes_obj[:, np.newaxis, :]),
                                   axis=1), sbj_k * rel_k * obj_k, axis=0))
                self.all_gt_labels[i].append(
                    np.concatenate((gt_labels_sbj[:, np.newaxis],
                                    gt_labels_rel[:, np.newaxis],
                                    gt_labels_obj[:, np.newaxis]),
                                   axis=1))
                self.all_gt_boxes[i].append(
                    np.concatenate((gt_boxes_sbj[:, np.newaxis, :],
                                    gt_boxes_obj[:, np.newaxis, :]),
                                   axis=1))

        return new_batch_flag

    def calculate_and_plot_accuracy(self):

        if True:
            self.tri_top1_acc = float(self.tri_top1_cnt) / float(self.spo_cnt) * 100
            self.tri_top5_acc = float(self.tri_top5_cnt) / float(self.spo_cnt) * 100
            self.tri_top10_acc = float(self.tri_top10_cnt) / float(self.spo_cnt) * 100
            self.sbj_top1_acc = float(self.sbj_top1_cnt) / float(self.spo_cnt) * 100
            self.sbj_top5_acc = float(self.sbj_top5_cnt) / float(self.spo_cnt) * 100
            self.sbj_top10_acc = float(self.sbj_top10_cnt) / float(self.spo_cnt) * 100
            self.obj_top1_acc = float(self.obj_top1_cnt) / float(self.spo_cnt) * 100
            self.obj_top5_acc = float(self.obj_top5_cnt) / float(self.spo_cnt) * 100
            self.obj_top10_acc = float(self.obj_top10_cnt) / float(self.spo_cnt) * 100
            self.rel_top1_acc = float(self.rel_top1_cnt) / float(self.spo_cnt) * 100
            self.rel_top5_acc = float(self.rel_top5_cnt) / float(self.spo_cnt) * 100
            self.rel_top10_acc = float(self.rel_top10_cnt) / float(self.spo_cnt) * 100

            if cfg.MODEL.WEAK_LABELS:
                self.tri_top1_w_acc = float(self.tri_top1_w_cnt) / float(self.spo_cnt) * 100
                self.sbj_top1_w_acc = float(self.sbj_top1_w_cnt) / float(self.spo_cnt) * 100
                self.obj_top1_w_acc = float(self.obj_top1_w_cnt) / float(self.spo_cnt) * 100
                self.rel_top1_w_acc = float(self.rel_top1_w_cnt) / float(self.spo_cnt) * 100

            self.sbj_mr /= float(self.spo_cnt) / 100
            self.rel_mr /= float(self.spo_cnt) / 100
            self.obj_mr /= float(self.spo_cnt) / 100
            self.tri_mr /= float(self.spo_cnt) / 100
            self.sbj_rr /= float(self.spo_cnt) / 100
            self.rel_rr /= float(self.spo_cnt) / 100
            self.obj_rr /= float(self.spo_cnt) / 100
            self.tri_rr /= float(self.spo_cnt) / 100

            print('triplet top 1 accuracy: {:f}'.format(self.tri_top1_acc))
            print('triplet top 5 accuracy: {:f}'.format(self.tri_top5_acc))
            print('triplet top 10 accuracy: {:f}'.format(self.tri_top10_acc))
            print('triplet rr: {:f}'.format(self.tri_rr))
            print('triplet mr: {:f}'.format(self.tri_mr))
            if cfg.MODEL.WEAK_LABELS:
                print('triplet top 1 accuracy weak labels: {:f}'.format(self.tri_top1_w_acc))

            print('sbj top 1 accuracy: {:f}'.format(self.sbj_top1_acc))
            print('sbj top 5 accuracy: {:f}'.format(self.sbj_top5_acc))
            print('sbj top 10 accuracy: {:f}'.format(self.sbj_top10_acc))
            print('sbj rr: {:f}'.format(self.sbj_rr))
            print('sbj mr: {:f}'.format(self.sbj_mr))
            if cfg.MODEL.WEAK_LABELS:
                print('sbj top 1 accuracy weak labels: {:f}'.format(self.sbj_top1_w_acc))

            print('obj top 1 accuracy: {:f}'.format(self.obj_top1_acc))
            print('obj top 5 accuracy: {:f}'.format(self.obj_top5_acc))
            print('obj top 10 accuracy: {:f}'.format(self.obj_top10_acc))
            print('obj rr: {:f}'.format(self.obj_rr))
            print('obj mr: {:f}'.format(self.obj_mr))
            if cfg.MODEL.WEAK_LABELS:
                print('obj top 1 accuracy weak labels: {:f}'.format(self.obj_top1_w_acc))

            print('rel top 1 accuracy: {:f}'.format(self.rel_top1_acc))
            print('rel top 5 accuracy: {:f}'.format(self.rel_top5_acc))
            print('rel top 10 accuracy: {:f}'.format(self.rel_top10_acc))
            print('rel rr: {:f}'.format(self.rel_rr))
            print('rel mr: {:f}'.format(self.rel_mr))
            if cfg.MODEL.WEAK_LABELS:
                print('rel top 1 accuracy weak labels: {:f}'.format(self.rel_top1_w_acc))

        all_accs = {}
        for key, val in self.__dict__.items():
            if key.find('acc') >= 0:
                all_accs[key] = val
        return all_accs

    def save_all_dets(self):

        output_dir = helpers_rel.get_output_directory()
        det_path = os.path.join(
            output_dir,
            cfg.DATASET,
            cfg.MODEL.TYPE, cfg.MODEL.SUBTYPE, cfg.MODEL.SPECS, cfg.TEST.DATA_TYPE)
        if not os.path.exists(det_path):
            os.makedirs(det_path)
        if cfg.MODEL.WEAK_LABELS:
            det_name = 'reldn_detections_w.pkl'
        else:
            det_name = 'reldn_detections.pkl'
        det_file = os.path.join(det_path, det_name)
        logger.info('all_dets size: {}'.format(len(self.all_dets['labels_sbj'])))
        with open(det_file, 'wb') as f:
            pickle.dump(self.all_dets, f, pickle.HIGHEST_PROTOCOL)
        logger.info('Wrote reldn detections to {}'.format(os.path.abspath(det_file)))

        if cfg.TEST.GET_ALL_LAN_EMBEDDINGS:
            all_obj_lan_embds_name = 'all_obj_lan_embds.pkl'
            all_obj_lan_embds_file = os.path.join(det_path, all_obj_lan_embds_name)
            logger.info('all_obj_lan_embds size: {}'.format(
                self.all_obj_lan_embds.shape[0]))
            with open(all_obj_lan_embds_file, 'wb') as f:
                pickle.dump(self.all_obj_lan_embds, f, pickle.HIGHEST_PROTOCOL)
            logger.info('Wrote all_obj_lan_embds to {}'.format(
                os.path.abspath(all_obj_lan_embds_file)))

            all_prd_lan_embds_name = 'all_prd_lan_embds.pkl'
            all_prd_lan_embds_file = os.path.join(det_path, all_prd_lan_embds_name)
            logger.info('all_prd_lan_embds size: {}'.format(
                self.all_prd_lan_embds.shape[0]))
            with open(all_prd_lan_embds_file, 'wb') as f:
                pickle.dump(self.all_prd_lan_embds, f, pickle.HIGHEST_PROTOCOL)
            logger.info('Wrote all_prd_lan_embds to {}'.format(
                os.path.abspath(all_prd_lan_embds_file)))

        if cfg.TEST.GET_ALL_VIS_EMBEDDINGS:
            all_sbj_vis_embds_name = 'all_sbj_vis_embds.pkl'
            all_sbj_vis_embds_file = os.path.join(det_path, all_sbj_vis_embds_name)
            logger.info('all_sbj_vis_embds size: {}'.format(
                len(self.all_sbj_vis_embds)))
            with open(all_sbj_vis_embds_file, 'wb') as f:
                pickle.dump(self.all_sbj_vis_embds, f, pickle.HIGHEST_PROTOCOL)
            logger.info('Wrote all_sbj_vis_embds to {}'.format(
                os.path.abspath(all_sbj_vis_embds_file)))

            all_obj_vis_embds_name = 'all_obj_vis_embds.pkl'
            all_obj_vis_embds_file = os.path.join(det_path, all_obj_vis_embds_name)
            logger.info('all_obj_vis_embds size: {}'.format(
                len(self.all_obj_vis_embds)))
            with open(all_obj_vis_embds_file, 'wb') as f:
                pickle.dump(self.all_obj_vis_embds, f, pickle.HIGHEST_PROTOCOL)
            logger.info('Wrote all_obj_vis_embds to {}'.format(
                os.path.abspath(all_obj_vis_embds_file)))

            all_prd_vis_embds_name = 'all_prd_vis_embds.pkl'
            all_prd_vis_embds_file = os.path.join(det_path, all_prd_vis_embds_name)
            logger.info('all_prd_vis_embds size: {}'.format(
                len(self.all_prd_vis_embds)))
            with open(all_prd_vis_embds_file, 'wb') as f:
                pickle.dump(self.all_prd_vis_embds, f, pickle.HIGHEST_PROTOCOL)
            logger.info('Wrote all_prd_vis_embds to {}'.format(
                os.path.abspath(all_prd_vis_embds_file)))

    def eval_im_dets_triplet_topk_enriched(self,
                                           image_idx,
                                           image_id,
                                           gt_labels_sbj,
                                           gt_labels_obj,
                                           gt_labels_rel,
                                           gt_boxes_sbj,
                                           gt_boxes_obj,
                                           gt_boxes_rel,
                                           det_boxes_sbj,
                                           det_boxes_obj,
                                           det_boxes_rel,
                                           det_labels_sbj,
                                           det_labels_obj,
                                           det_labels_rel,
                                           det_scores_sbj,
                                           det_scores_obj,
                                           det_scores_rel, detections_results_keys):

        detections_results_keys = ['tri_top1', 'tri_top5', 'tri_top10',
                                   'sbj_top1', 'sbj_top5', 'sbj_top10',
                                   'obj_top1', 'obj_top5', 'obj_top10',
                                   'rel_top1', 'rel_top5', 'rel_top10',
                                   'tri_rr', 'sbj_rr', 'obj_rr', 'rel_rr',
                                   'tri_mr', 'sbj_mr', 'obj_mr', 'rel_mr']

        detections_results_image = {key: np.zeros((len(gt_labels_sbj), 1)) for key in detections_results_keys}

        #         tri_top1 = np.zeros(( len(gt_labels_sbj),1))
        #         tri_top5 = np.zeros(( len(gt_labels_sbj),1))
        #         tri_top10 = np.zeros(( len(gt_labels_sbj),1))

        #         sbj_top1 = np.zeros(( len(gt_labels_sbj),1))
        #         sbj_top5 = np.zeros(( len(gt_labels_sbj),1))
        #         sbj_top10 = np.zeros(( len(gt_labels_sbj),1))

        #         obj_top1 = np.zeros(( len(gt_labels_sbj),1))
        #         obj_top5 = np.zeros(( len(gt_labels_sbj),1))
        #         obj_top10 = np.zeros(( len(gt_labels_sbj),1))

        #         rel_top1 = np.zeros(( len(gt_labels_sbj),1))
        #         rel_top5 = np.zeros(( len(gt_labels_sbj),1))
        #         rel_top10 = np.zeros(( len(gt_labels_sbj),1))

        #         tri_rr = np.zeros(( len(gt_labels_sbj),1))
        #         sbj_rr = np.zeros(( len(gt_labels_sbj),1))
        #         obj_rr = np.zeros(( len(gt_labels_sbj),1))
        #         rel_rr = np.zeros(( len(gt_labels_sbj),1))

        #         tri_mr = np.zeros(( len(gt_labels_sbj),1))
        #         sbj_mr = np.zeros(( len(gt_labels_sbj),1))
        #         obj_mr = np.zeros(( len(gt_labels_sbj),1))
        #         rel_mr = np.zeros(( len(gt_labels_sbj),1))
        detections_results_keys = ['tri_top1', 'tri_top5', 'tri_top10',
                                   'sbj_top1', 'sbj_top5', 'sbj_top10',
                                   'obj_top1', 'obj_top5', 'obj_top10',
                                   'rel_top1', 'rel_top5', 'rel_top10',
                                   'tri_rr', 'sbj_rr', 'obj_rr', 'rel_rr',
                                   'tri_mr', 'sbj_mr', 'obj_mr', 'rel_mr']

        self.spo_cnt += len(gt_labels_sbj)
        for ind in range(len(gt_labels_sbj)):
            if gt_labels_sbj[ind] in det_labels_sbj[ind, :1] and \
                    gt_labels_obj[ind] in det_labels_obj[ind, :1] and \
                    gt_labels_rel[ind] in det_labels_rel[ind, :1]:
                detections_results_image['tri_top1'][ind] = 1
                self.tri_top1_cnt += 1
            if gt_labels_sbj[ind] in det_labels_sbj[ind, :1]:
                self.sbj_top1_cnt += 1
                detections_results_image['sbj_top1'][ind] = 1
            if gt_labels_obj[ind] in det_labels_obj[ind, :1]:
                self.obj_top1_cnt += 1
                detections_results_image['obj_top1'][ind] = 1
            if gt_labels_rel[ind] in det_labels_rel[ind, :1]:
                self.rel_top1_cnt += 1
                detections_results_image['rel_top1'][ind] = 1

            if gt_labels_sbj[ind] in det_labels_sbj[ind, :5] and \
                    gt_labels_obj[ind] in det_labels_obj[ind, :5] and \
                    gt_labels_rel[ind] in det_labels_rel[ind, :5]:
                self.tri_top5_cnt += 1
                detections_results_image['tri_top5'][ind] = 1
            if gt_labels_sbj[ind] in det_labels_sbj[ind, :5]:
                self.sbj_top5_cnt += 1
                detections_results_image['sbj_top5'][ind] = 1
            if gt_labels_obj[ind] in det_labels_obj[ind, :5]:
                self.obj_top5_cnt += 1
                detections_results_image['obj_top5'][ind] = 1
            if gt_labels_rel[ind] in det_labels_rel[ind, :5]:
                self.rel_top5_cnt += 1
                detections_results_image['rel_top5'][ind] = 1

            if gt_labels_sbj[ind] in det_labels_sbj[ind, :10] and \
                    gt_labels_obj[ind] in det_labels_obj[ind, :10] and \
                    gt_labels_rel[ind] in det_labels_rel[ind, :10]:
                self.tri_top10_cnt += 1
                detections_results_image['tri_top10'][ind] = 1
            if gt_labels_sbj[ind] in det_labels_sbj[ind, :10]:
                self.sbj_top10_cnt += 1
                detections_results_image['sbj_top10'][ind] = 1
            if gt_labels_obj[ind] in det_labels_obj[ind, :10]:
                self.obj_top10_cnt += 1
                detections_results_image['obj_top10'][ind] = 1
            if gt_labels_rel[ind] in det_labels_rel[ind, :10]:
                self.rel_top10_cnt += 1
                detections_results_image['rel_top10'][ind] = 1

            s_correct = gt_labels_sbj[ind] in det_labels_sbj[ind, :self.rank_k]
            p_correct = gt_labels_rel[ind] in det_labels_rel[ind, :self.rank_k]
            o_correct = gt_labels_obj[ind] in det_labels_obj[ind, :self.rank_k]
            spo_correct = s_correct and p_correct and o_correct
            s_ind = np.where(
                det_labels_sbj[ind, :self.rank_k].squeeze() == \
                gt_labels_sbj[ind])[0]
            p_ind = np.where(
                det_labels_rel[ind, :self.rank_k].squeeze() == \
                gt_labels_rel[ind])[0]
            o_ind = np.where(
                det_labels_obj[ind, :self.rank_k].squeeze() == \
                gt_labels_obj[ind])[0]

            self.sbj_mr += 1
            self.rel_mr += 1
            self.obj_mr += 1
            self.tri_mr += 1
            if s_correct:
                s_ind = s_ind[0]
                self.sbj_rr += 1.0 / (s_ind + 1.0)
                detections_results_image['sbj_rr'][ind] = 1.0 / (s_ind + 1.0)
                self.sbj_mr += s_ind / float(self.rank_k) - 1
                detections_results_image['sbj_mr'][ind] = s_ind / float(self.rank_k) - 1
            if p_correct:
                p_ind = p_ind[0]
                self.rel_rr += 1.0 / (p_ind + 1.0)
                detections_results_image['rel_rr'][ind] = 1.0 / (p_ind + 1.0)
                self.rel_mr += p_ind / float(self.rank_k) - 1
                detections_results_image['rel_mr'][ind] = p_ind / float(self.rank_k) - 1
            if o_correct:
                o_ind = o_ind[0]
                self.obj_rr += 1.0 / (o_ind + 1.0)
                detections_results_image['obj_rr'][ind] = 1.0 / (o_ind + 1.0)
                self.obj_mr += o_ind / float(self.rank_k) - 1
                detections_results_image['obj_mr'][ind] = o_ind / float(self.rank_k) - 1
            if spo_correct:
                self.tri_rr += (1.0 / (s_ind + 1.0) + \
                                1.0 / (p_ind + 1.0) + \
                                1.0 / (o_ind + 1.0)) / 3.0

                detections_results_image['tri_rr'][ind] = (1.0 / (s_ind + 1.0) + \
                                                           1.0 / (p_ind + 1.0) + \
                                                           1.0 / (o_ind + 1.0)) / 3.0

                self.tri_mr += (s_ind / float(self.rank_k) - 1 + \
                                p_ind / float(self.rank_k) - 1 + \
                                o_ind / float(self.rank_k) - 1) / 3.0

                detections_results_image['tri_mr'][ind] = (s_ind / float(self.rank_k) - 1 + \
                                                           p_ind / float(self.rank_k) - 1 + \
                                                           o_ind / float(self.rank_k) - 1) / 3.0

        sbj_k = 1
        # rel_k = 70
        obj_k = 1
        # det_labels = []
        # det_boxes = []
        # gt_labels = []
        # gt_boxes = []
        for i, rel_k in enumerate(self.all_rel_k):
            if det_labels_sbj.shape[0] > 0:
                topk_labels_sbj = det_labels_sbj[:, :sbj_k]
                topk_labels_rel = det_labels_rel[:, :rel_k]
                topk_labels_obj = det_labels_obj[:, :obj_k]
            else:  # In the ECCV2016 proposals sometimes there is no det box
                topk_labels_sbj = np.zeros((0, sbj_k), dtype=np.int32)
                topk_labels_rel = np.zeros((0, rel_k), dtype=np.int32)
                topk_labels_obj = np.zeros((0, obj_k), dtype=np.int32)

            if det_scores_sbj.shape[0] > 0:
                topk_scores_sbj = det_scores_sbj[:, :sbj_k]
                topk_scores_rel = det_scores_rel[:, :rel_k]
                topk_scores_obj = det_scores_obj[:, :obj_k]
            else:  # In the ECCV2016 proposals sometimes there is no det box
                topk_scores_sbj = np.zeros((0, sbj_k), dtype=np.float32)
                topk_scores_rel = np.zeros((0, rel_k), dtype=np.float32)
                topk_scores_obj = np.zeros((0, obj_k), dtype=np.float32)

            topk_cube_spo_labels = np.zeros(
                (topk_labels_sbj.shape[0], sbj_k * obj_k * rel_k, 3), dtype=np.int32)
            topk_cube_spo_scores = np.zeros(
                (topk_labels_sbj.shape[0], sbj_k * obj_k * rel_k), dtype=np.float32)
            topk_cube_p_scores = np.zeros(
                (topk_labels_sbj.shape[0], sbj_k * obj_k * rel_k), dtype=np.float32)
            for l in range(sbj_k):
                for m in range(rel_k):
                    for n in range(obj_k):
                        topk_cube_spo_labels[:, l * rel_k * obj_k + m * obj_k + n, 0] = \
                            topk_labels_sbj[:, l]
                        topk_cube_spo_labels[:, l * rel_k * obj_k + m * obj_k + n, 1] = \
                            topk_labels_rel[:, m]
                        topk_cube_spo_labels[:, l * rel_k * obj_k + m * obj_k + n, 2] = \
                            topk_labels_obj[:, n]
                        topk_cube_spo_scores[:, l * rel_k * obj_k + m * obj_k + n] = \
                            np.exp(topk_scores_sbj[:, l] +
                                   topk_scores_rel[:, m] +
                                   topk_scores_obj[:, n])

                        topk_cube_p_scores[:, l * rel_k * obj_k + m * obj_k + n] = \
                            np.exp(topk_scores_rel[:, m])

            topk_cube_spo_labels_reshape = topk_cube_spo_labels.reshape((-1, 3))
            topk_cube_spo_scores_reshape = topk_cube_spo_scores.reshape((-1, 1))

            self.all_det_labels[i].append(
                np.concatenate((topk_cube_spo_scores_reshape[:, 0, np.newaxis],
                                topk_cube_spo_labels_reshape[:, 0, np.newaxis],
                                topk_cube_spo_labels_reshape[:, 1, np.newaxis],
                                topk_cube_spo_labels_reshape[:, 2, np.newaxis]),
                               axis=1))
            self.all_det_boxes[i].append(np.repeat(
                np.concatenate((det_boxes_sbj[:, np.newaxis, :],
                                det_boxes_obj[:, np.newaxis, :]),
                               axis=1), sbj_k * rel_k * obj_k, axis=0))
            self.all_gt_labels[i].append(
                np.concatenate((gt_labels_sbj[:, np.newaxis],
                                gt_labels_rel[:, np.newaxis],
                                gt_labels_obj[:, np.newaxis]),
                               axis=1))
            self.all_gt_boxes[i].append(
                np.concatenate((gt_boxes_sbj[:, np.newaxis, :],
                                gt_boxes_obj[:, np.newaxis, :]),
                               axis=1))

        return detections_results_image
