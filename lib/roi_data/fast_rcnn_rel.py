# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import numpy.random as npr
import math
import pickle

from core.config_rel import cfg
import utils.boxes as box_utils

from utils.timer import Timer

import logging

logger = logging.getLogger(__name__)


def add_fast_rcnn_blobs(
        blobs, im_scales, landb, roidb, roidb_inds, proposals, split, low_shot_helper):
    """Add blobs needed for training Fast R-CNN style models."""
    # Sample training RoIs from each image and append them to the blob lists
    if split != 'train':
        assert proposals is not None, \
            'proposals should not be None during val/test.'
        for im_i, entry in enumerate(roidb):
            scale = im_scales[im_i]
            # labels start from 1
            sbj_gt_labels = entry['sbj_max_classes']
            obj_gt_labels = entry['obj_max_classes']
            rel_gt_labels = entry['rel_max_classes']

            if cfg.MODEL.WEAK_LABELS:
                sbj_gt_labels_w = entry['sbj_max_classes_w']
                obj_gt_labels_w = entry['obj_max_classes_w']
                rel_gt_labels_w = entry['rel_max_classes_w']
            sbj_gt_boxes = entry['sbj_boxes'] * scale
            obj_gt_boxes = entry['obj_boxes'] * scale
            rel_gt_boxes = entry['rel_boxes'] * scale

            num_proposals = proposals[im_i]['boxes_sbj'].shape[0]
            # logger.info('num_proposals: {}'.format(num_proposals))
            if num_proposals > 0:
                all_sbj_rois = np.zeros((num_proposals, 5), dtype=np.float32)
                all_obj_rois = np.zeros((num_proposals, 5), dtype=np.float32)
                all_rel_rois = np.zeros((num_proposals, 5), dtype=np.float32)
                all_sbj_rois[:, 1:5] = proposals[im_i]['boxes_sbj'] * scale
                all_obj_rois[:, 1:5] = proposals[im_i]['boxes_obj'] * scale
                all_rel_rois[:, 1:5] = proposals[im_i]['boxes_rel'] * scale
            else:  # create dummy rois
                all_sbj_rois = np.zeros((1, 5), dtype=np.float32)
                all_obj_rois = np.zeros((1, 5), dtype=np.float32)
                all_rel_rois = np.zeros((1, 5), dtype=np.float32)
                all_sbj_rois[:, 3:5] = 1.0
                all_obj_rois[:, 3:5] = 1.0
                all_rel_rois[:, 3:5] = 1.0

            subbatch_id = proposals[im_i]['subbatch_id']

            frcn_blobs = {}
            frcn_blobs['sbj_rois'] = all_sbj_rois
            frcn_blobs['obj_rois'] = all_obj_rois
            frcn_blobs['rel_rois_sbj'] = all_sbj_rois
            frcn_blobs['rel_rois_obj'] = all_obj_rois
            frcn_blobs['rel_rois_prd'] = all_rel_rois
            frcn_blobs['sbj_pos_labels_int32'] = sbj_gt_labels.astype(np.int32)
            frcn_blobs['obj_pos_labels_int32'] = obj_gt_labels.astype(np.int32)
            frcn_blobs['rel_pos_labels_int32'] = rel_gt_labels.astype(np.int32)
            if cfg.MODEL.WEAK_LABELS:
                for num_w in range(cfg.MODEL.NUM_WEAK_LABELS):
                    frcn_blobs['sbj_pos_labels_int32_w_' + str(num_w)] = sbj_gt_labels_w[:, num_w].astype(np.int32,
                                                                                                          copy=False)  # weak labels
                    frcn_blobs['obj_pos_labels_int32_w_' + str(num_w)] = obj_gt_labels_w[:, num_w].astype(np.int32,
                                                                                                          copy=False)  # weak labels
                    frcn_blobs['rel_pos_labels_int32_w_' + str(num_w)] = rel_gt_labels_w[:, num_w].astype(np.int32,
                                                                                                          copy=False)  # weak labels

                frcn_blobs['sbj_pos_labels_float32_w'] = np.zeros((sbj_gt_labels_w.shape[0], cfg.MODEL.NUM_CLASSES_SBJ_OBJ),
                                                            dtype=np.float32)
                frcn_blobs['obj_pos_labels_float32_w'] = np.zeros((obj_gt_labels_w.shape[0], cfg.MODEL.NUM_CLASSES_SBJ_OBJ),
                                                            dtype=np.float32)
                frcn_blobs['rel_pos_labels_float32_w'] = np.zeros((rel_gt_labels_w.shape[0], cfg.MODEL.NUM_CLASSES_PRD),
                                                            dtype=np.float32)

                overlap_sbj = np.array(
                    [(sbj_gt_labels[i] in sbj_gt_labels_w[i, :]) for i in range(len(sbj_gt_labels))])
                # overlap_sbj = np.isin(sbj_gt_labels, sbj_gt_labels_w)
                overlap_obj = np.array(
                    [(obj_gt_labels[i] in obj_gt_labels_w[i, :]) for i in range(len(obj_gt_labels))])
                # overlap_obj = np.isin(obj_gt_labels, obj_gt_labels_w)
                overlap_rel = np.array(
                    [(rel_gt_labels[i] in rel_gt_labels_w[i, :]) for i in range(len(rel_gt_labels))])
                # overlap_rel = np.isin(rel_gt_labels, rel_gt_labels_w)
                denominator_sbj = np.zeros((sbj_gt_labels_w.shape[0],), dtype=np.float32)
                denominator_obj = np.zeros((obj_gt_labels_w.shape[0],), dtype=np.float32)
                denominator_rel = np.zeros((rel_gt_labels_w.shape[0],), dtype=np.float32)

                denominator_sbj[np.where(overlap_sbj == False)] = 1.0 / (cfg.MODEL.NUM_WEAK_LABELS + 1)
                denominator_sbj[np.where(overlap_sbj == True)] = 1.0 / cfg.MODEL.NUM_WEAK_LABELS

                denominator_obj[np.where(overlap_obj == False)] = 1.0 / (cfg.MODEL.NUM_WEAK_LABELS + 1)
                denominator_obj[np.where(overlap_obj == True)] = 1.0 / cfg.MODEL.NUM_WEAK_LABELS

                denominator_rel[np.where(overlap_rel == False)] = 1.0 / (cfg.MODEL.NUM_WEAK_LABELS + 1)
                denominator_rel[np.where(overlap_rel == True)] = 1.0 / cfg.MODEL.NUM_WEAK_LABELS

                # while True:
                #	print('sbj_gt_labels_w', sbj_gt_labels_w)

                idx_batch_sbj = np.arange(sbj_gt_labels_w.shape[0], dtype=np.int32)
                idx_batch_obj = np.arange(obj_gt_labels_w.shape[0], dtype=np.int32)
                idx_batch_rel = np.arange(rel_gt_labels_w.shape[0], dtype=np.int32)
                # idx = np.stack([idx_batch, sbj_gt_labels_w[:, 1].astype(np.int32)], axis=1)


                for num_w in range(cfg.MODEL.NUM_WEAK_LABELS):
                    idx_sbj = sbj_gt_labels_w[:, num_w].astype(np.int32)
                    idx_obj = obj_gt_labels_w[:, num_w].astype(np.int32)
                    idx_rel = rel_gt_labels_w[:, num_w].astype(np.int32)

                    frcn_blobs['sbj_pos_labels_float32_w'][idx_batch_sbj, idx_sbj] = denominator_sbj
                    frcn_blobs['obj_pos_labels_float32_w'][idx_batch_obj, idx_obj] = denominator_obj
                    frcn_blobs['rel_pos_labels_float32_w'][idx_batch_rel, idx_rel] = denominator_rel

                idx_sbj = sbj_gt_labels.astype(np.int32)
                idx_obj = obj_gt_labels.astype(np.int32)
                idx_rel = rel_gt_labels.astype(np.int32)

                frcn_blobs['sbj_pos_labels_float32_w'][idx_batch_sbj, idx_sbj] = denominator_sbj
                frcn_blobs['obj_pos_labels_float32_w'][idx_batch_obj, idx_obj] = denominator_obj
                frcn_blobs['rel_pos_labels_float32_w'][idx_batch_rel, idx_rel] = denominator_rel

                # frcn_blobs['sbj_pos_labels_float32_w'] = np.zeros(sbj_gt_labels_w[:, 0].shape, dtype=np.float32)
                # frcn_blobs['obj_pos_labels_float32_w'] = np.zeros(obj_gt_labels_w[:, 0].shape, dtype=np.float32)
                # frcn_blobs['rel_pos_labels_float32_w'] = np.zeros(rel_gt_labels_w[:, 0].shape, dtype=np.float32)
                #
                # if np.argmax(sbj_gt_labels) in [np.argmax(sbj_gt_labels_w[:, num_w]) for num_w in
                #                                 range(cfg.MODEL.NUM_WEAK_LABELS)]:
                #     denominator_sbj = 1.0 / cfg.MODEL.NUM_WEAK_LABELS
                # else:
                #     denominator_sbj = 1.0 / (cfg.MODEL.NUM_WEAK_LABELS + 1)
                #
                # if np.argmax(obj_gt_labels) in [np.argmax(obj_gt_labels_w[:, num_w]) for num_w in
                #                                 range(cfg.MODEL.NUM_WEAK_LABELS)]:
                #     denominator_obj = 1.0 / cfg.MODEL.NUM_WEAK_LABELS
                # else:
                #     denominator_obj = 1.0 / (cfg.MODEL.NUM_WEAK_LABELS + 1)
                #
                # if np.argmax(rel_gt_labels) in [np.argmax(rel_gt_labels_w[:, num_w]) for num_w in
                #                                 range(cfg.MODEL.NUM_WEAK_LABELS)]:
                #     denominator_rel = 1.0 / cfg.MODEL.NUM_WEAK_LABELS
                # else:
                #     denominator_rel = 1.0 / (cfg.MODEL.NUM_WEAK_LABELS + 1)
                #
                # for num_w in range(cfg.MODEL.NUM_WEAK_LABELS):
                #     frcn_blobs['sbj_pos_labels_float32_w'][np.argmax(sbj_gt_labels_w[:, num_w])] = 1.0 / denominator_sbj
                #     frcn_blobs['obj_pos_labels_float32_w'][np.argmax(obj_gt_labels_w[:, num_w])] = 1.0 / denominator_obj
                #     frcn_blobs['rel_pos_labels_float32_w'][np.argmax(rel_gt_labels_w[:, num_w])] = 1.0 / denominator_rel
                #
                # frcn_blobs['sbj_pos_labels_float32_w'][np.argmax(sbj_gt_labels)] = 1.0 / denominator_sbj
                # frcn_blobs['obj_pos_labels_float32_w'][np.argmax(obj_gt_labels)] = 1.0 / denominator_obj
                # frcn_blobs['rel_pos_labels_float32_w'][np.argmax(rel_gt_labels)] = 1.0 / denominator_rel

            frcn_blobs['sbj_gt_boxes'] = sbj_gt_boxes.astype(np.float32)
            frcn_blobs['obj_gt_boxes'] = obj_gt_boxes.astype(np.float32)
            frcn_blobs['rel_gt_boxes'] = rel_gt_boxes.astype(np.float32)
            frcn_blobs['image_idx'] = np.array(roidb_inds[im_i])[np.newaxis].astype(np.int32)
            frcn_blobs['image_id'] = \
                np.array(float(entry['image'].split('/')[-1][:-4]))[np.newaxis].astype(np.float32)
            frcn_blobs['image_scale'] = np.array(scale)[np.newaxis].astype(np.float32)
            frcn_blobs['subbatch_id'] = \
                np.array(subbatch_id)[np.newaxis].astype(np.float32)
            frcn_blobs['num_proposals'] = \
                np.array(num_proposals)[np.newaxis].astype(np.float32)

            for k, v in frcn_blobs.items():
                blobs[k].append(v)
        # Concat the training blob lists into tensors
        for k, v in blobs.items():
            if isinstance(v, list) and len(v) > 0:
                blobs[k] = np.concatenate(v)
        return True
    else:
        for im_i, entry in enumerate(roidb):
            scale = im_scales[im_i]
            sbj_gt_inds = np.where((entry['gt_sbj_classes'] > 0))[0]
            obj_gt_inds = np.where((entry['gt_obj_classes'] > 0))[0]

            sbj_gt_rois = entry['sbj_boxes'][sbj_gt_inds, :] * scale
            obj_gt_rois = entry['obj_boxes'][obj_gt_inds, :] * scale

            sbj_gt_rois = sbj_gt_rois.astype(np.float32)
            obj_gt_rois = obj_gt_rois.astype(np.float32)

            sbj_gt_boxes = np.zeros((len(sbj_gt_inds), 6), dtype=np.float32)
            sbj_gt_boxes[:, 0] = im_i  # batch inds
            sbj_gt_boxes[:, 1:5] = sbj_gt_rois
            sbj_gt_boxes[:, 5] = entry['gt_sbj_classes'][sbj_gt_inds]

            obj_gt_boxes = np.zeros((len(obj_gt_inds), 6), dtype=np.float32)
            obj_gt_boxes[:, 0] = im_i  # batch inds
            obj_gt_boxes[:, 1:5] = obj_gt_rois
            obj_gt_boxes[:, 5] = entry['gt_obj_classes'][obj_gt_inds]

            # labels start from 1
            sbj_gt_labels = entry['sbj_max_classes']
            obj_gt_labels = entry['obj_max_classes']
            rel_gt_labels = entry['rel_max_classes']

            if cfg.MODEL.WEAK_LABELS:
                sbj_gt_labels_w = entry['sbj_max_classes_w']
                obj_gt_labels_w = entry['obj_max_classes_w']
                rel_gt_labels_w = entry['rel_max_classes_w']

            sbj_gt_vecs = entry['sbj_vecs']
            obj_gt_vecs = entry['obj_vecs']
            rel_gt_vecs = entry['prd_vecs']

            if proposals is None:
                # Get unique boxes
                rows = set()
                unique_sbj_gt_inds = []
                for idx, row in enumerate(sbj_gt_boxes):
                    if tuple(row) not in rows:
                        rows.add(tuple(row))
                        unique_sbj_gt_inds.append(idx)
                unique_sbj_gt_boxes = sbj_gt_boxes[unique_sbj_gt_inds, :]

                rows = set()
                unique_obj_gt_inds = []
                for idx, row in enumerate(obj_gt_boxes):
                    if tuple(row) not in rows:
                        rows.add(tuple(row))
                        unique_obj_gt_inds.append(idx)
                unique_obj_gt_boxes = obj_gt_boxes[unique_obj_gt_inds, :]

                # use better sampling by default
                im_width = entry['width'] * scale
                im_height = entry['height'] * scale

                _rois_sbj = _augment_gt_boxes_by_perturbation(
                    unique_sbj_gt_boxes[:, 1:5], im_width, im_height)
                rois_sbj = np.zeros((_rois_sbj.shape[0], 5), dtype=np.float32)
                rois_sbj[:, 0] = im_i
                rois_sbj[:, 1:5] = _rois_sbj

                _rois_obj = _augment_gt_boxes_by_perturbation(
                    unique_obj_gt_boxes[:, 1:5], im_width, im_height)
                rois_obj = np.zeros((_rois_obj.shape[0], 5), dtype=np.float32)
                rois_obj[:, 0] = im_i
                rois_obj[:, 1:5] = _rois_obj

                rows = set()
                unique_sbj_rois_inds = []
                for idx, row in enumerate(rois_sbj):
                    if tuple(row) not in rows:
                        rows.add(tuple(row))
                        unique_sbj_rois_inds.append(idx)
                unique_rois_sbj = rois_sbj[unique_sbj_rois_inds, :]

                rows = set()
                unique_obj_rois_inds = []
                for idx, row in enumerate(rois_obj):
                    if tuple(row) not in rows:
                        rows.add(tuple(row))
                        unique_obj_rois_inds.append(idx)
                unique_rois_obj = rois_obj[unique_obj_rois_inds, :]

                unique_all_rois_sbj = \
                    np.vstack((unique_rois_sbj, unique_sbj_gt_boxes[:, :-1]))
                unique_all_rois_obj = \
                    np.vstack((unique_rois_obj, unique_obj_gt_boxes[:, :-1]))

                sbj_gt_boxes = sbj_gt_boxes[:, 1:]  # strip off batch index
                obj_gt_boxes = obj_gt_boxes[:, 1:]

                unique_sbj_gt_boxes = unique_sbj_gt_boxes[:, 1:]  # strip off batch index
                unique_obj_gt_boxes = unique_obj_gt_boxes[:, 1:]

                unique_sbj_gt_vecs = sbj_gt_vecs[unique_sbj_gt_inds]
                unique_obj_gt_vecs = obj_gt_vecs[unique_obj_gt_inds]

                unique_sbj_gt_labels = sbj_gt_labels[unique_sbj_gt_inds]
                unique_obj_gt_labels = obj_gt_labels[unique_obj_gt_inds]
            else:
                unique_all_rois_sbj = proposals[im_i]['unique_all_rois_sbj'] * scale
                unique_all_rois_obj = proposals[im_i]['unique_all_rois_obj'] * scale

                unique_all_rois_sbj[:, 0] = im_i
                unique_all_rois_obj[:, 0] = im_i

                unique_sbj_gt_inds = proposals[im_i]['unique_sbj_gt_inds']
                unique_obj_gt_inds = proposals[im_i]['unique_obj_gt_inds']

                sbj_gt_boxes = sbj_gt_boxes[:, 1:]  # strip off batch index
                obj_gt_boxes = obj_gt_boxes[:, 1:]

                unique_sbj_gt_boxes = sbj_gt_boxes[unique_sbj_gt_inds, :]
                unique_obj_gt_boxes = obj_gt_boxes[unique_obj_gt_inds, :]

                unique_sbj_gt_vecs = sbj_gt_vecs[unique_sbj_gt_inds]
                unique_obj_gt_vecs = obj_gt_vecs[unique_obj_gt_inds]

                unique_sbj_gt_labels = sbj_gt_labels[unique_sbj_gt_inds]
                unique_obj_gt_labels = obj_gt_labels[unique_obj_gt_inds]

                if cfg.MODEL.WEAK_LABELS:
                    unique_sbj_gt_labels_w = sbj_gt_labels_w[unique_sbj_gt_inds]
                    unique_obj_gt_labels_w = obj_gt_labels_w[unique_obj_gt_inds]

            if cfg.MODEL.LOSS_TYPE.find('TRIPLET') >= 0:
                if cfg.MODEL.WEAK_LABELS:
                    frcn_blobs = _sample_rois_triplet_yall(
                        unique_all_rois_sbj, unique_all_rois_obj,
                        unique_sbj_gt_boxes, unique_obj_gt_boxes,
                        unique_sbj_gt_vecs, unique_obj_gt_vecs,
                        unique_sbj_gt_labels, unique_obj_gt_labels,
                        sbj_gt_boxes, obj_gt_boxes,
                        sbj_gt_vecs, obj_gt_vecs, rel_gt_vecs,
                        rel_gt_labels,
                        low_shot_helper,
                        unique_sbj_gt_labels_w, unique_obj_gt_labels_w, rel_gt_labels_w)
                else:
                    frcn_blobs = _sample_rois_triplet_yall(
                        unique_all_rois_sbj, unique_all_rois_obj,
                        unique_sbj_gt_boxes, unique_obj_gt_boxes,
                        unique_sbj_gt_vecs, unique_obj_gt_vecs,
                        unique_sbj_gt_labels, unique_obj_gt_labels,
                        sbj_gt_boxes, obj_gt_boxes,
                        sbj_gt_vecs, obj_gt_vecs, rel_gt_vecs,
                        rel_gt_labels,
                        low_shot_helper)

            elif cfg.MODEL.LOSS_TYPE == 'SOFTMAX':
                frcn_blobs = _sample_rois_softmax_yall(
                    unique_all_rois_sbj, unique_all_rois_obj,
                    unique_sbj_gt_boxes, unique_obj_gt_boxes,
                    unique_sbj_gt_vecs, unique_obj_gt_vecs,
                    unique_sbj_gt_labels, unique_obj_gt_labels,
                    sbj_gt_boxes, obj_gt_boxes,
                    sbj_gt_vecs, obj_gt_vecs, rel_gt_vecs,
                    rel_gt_labels,
                    low_shot_helper)
            else:
                raise KeyError('Unknown loss type: {}'.format(cfg.MODEL.LOSS_TYPE))

            for k, v in frcn_blobs.items():
                blobs[k].append(v)
        # Concat the training blob lists into tensors
        for k, v in blobs.items():
            if isinstance(v, list) and len(v) > 0:
                blobs[k] = np.concatenate(v)
        return True


def _augment_gt_boxes_by_perturbation(unique_gt_boxes, im_width, im_height):
    num_gt = unique_gt_boxes.shape[0]
    num_rois = 1000
    rois = np.zeros((num_rois, 4), dtype=np.float32)
    cnt = 0
    for i in range(num_gt):
        box = unique_gt_boxes[i]
        box_width = box[2] - box[0] + 1
        box_height = box[3] - box[1] + 1
        x_offset_max = (box_width - 1) // 2
        y_offset_max = (box_height - 1) // 2
        for _ in range(num_rois // num_gt):
            x_min_offset = np.random.uniform(low=-x_offset_max, high=x_offset_max)
            y_min_offset = np.random.uniform(low=-y_offset_max, high=y_offset_max)
            x_max_offset = np.random.uniform(low=-x_offset_max, high=x_offset_max)
            y_max_offset = np.random.uniform(low=-y_offset_max, high=y_offset_max)

            new_x_min = min(max(np.round(box[0] + x_min_offset), 0), im_width - 1)
            new_y_min = min(max(np.round(box[1] + y_min_offset), 0), im_height - 1)
            new_x_max = min(max(np.round(box[2] + x_max_offset), 0), im_width - 1)
            new_y_max = min(max(np.round(box[3] + y_max_offset), 0), im_height - 1)

            new_box = np.array(
                [new_x_min, new_y_min, new_x_max, new_y_max]).astype(np.float32)
            rois[cnt] = new_box
            cnt += 1

    return rois


def _sample_rois_triplet_yall(
        unique_all_rois_sbj, unique_all_rois_obj,
        unique_sbj_gt_boxes, unique_obj_gt_boxes,
        unique_sbj_gt_vecs, unique_obj_gt_vecs,
        unique_sbj_gt_labels, unique_obj_gt_labels,
        sbj_gt_boxes, obj_gt_boxes,
        sbj_gt_vecs, obj_gt_vecs,
        rel_gt_vecs, rel_gt_labels,
        low_shot_helper,
        unique_sbj_gt_labels_w=None, unique_obj_gt_labels_w=None, rel_gt_labels_w=None):
    if cfg.TRAIN.OVERSAMPLE_SO:
        if cfg.MODEL.WEAK_LABELS:
            rois_sbj, pos_vecs_sbj, all_labels_sbj, all_labels_sbj_w, \
            neg_affinity_mask_sbj, pos_affinity_mask_sbj, \
            low_shot_ends_sbj, regular_starts_sbj = \
                _sample_rois_pos_neg_for_one_branch(
                    unique_all_rois_sbj, unique_sbj_gt_boxes,
                    unique_sbj_gt_labels, unique_sbj_gt_vecs,
                    low_shot_helper, 'sbj', unique_sbj_gt_labels_w)
        else:
            rois_sbj, pos_vecs_sbj, all_labels_sbj, \
            neg_affinity_mask_sbj, pos_affinity_mask_sbj, \
            low_shot_ends_sbj, regular_starts_sbj = \
                _sample_rois_pos_neg_for_one_branch(
                    unique_all_rois_sbj, unique_sbj_gt_boxes,
                    unique_sbj_gt_labels, unique_sbj_gt_vecs,
                    low_shot_helper, 'sbj')
    else:
        if cfg.MODEL.WEAK_LABELS:
            rois_sbj, pos_vecs_sbj, all_labels_sbj, all_labels_sbj_w, \
            neg_affinity_mask_sbj, pos_affinity_mask_sbj = \
                _sample_rois_pos_neg_for_one_branch(
                    unique_all_rois_sbj, unique_sbj_gt_boxes,
                    unique_sbj_gt_labels, unique_sbj_gt_vecs,
                    low_shot_helper, 'sbj', unique_sbj_gt_labels_w)
        else:
            rois_sbj, pos_vecs_sbj, all_labels_sbj, \
            neg_affinity_mask_sbj, pos_affinity_mask_sbj = \
                _sample_rois_pos_neg_for_one_branch(
                    unique_all_rois_sbj, unique_sbj_gt_boxes,
                    unique_sbj_gt_labels, unique_sbj_gt_vecs,
                    low_shot_helper, 'sbj')
    fg_size_sbj = pos_vecs_sbj.shape[0]
    pos_starts_sbj = np.array([0, 0], dtype=np.int32)
    pos_ends_sbj = np.array([fg_size_sbj, -1], dtype=np.int32)
    neg_starts_sbj = np.array([fg_size_sbj, 0], dtype=np.int32)
    neg_ends_sbj = np.array([-1, -1], dtype=np.int32)
    sbj_pos_labels = all_labels_sbj[:fg_size_sbj] - 1

    if cfg.MODEL.WEAK_LABELS:
        sbj_pos_labels_w = all_labels_sbj_w[:fg_size_sbj] - 1  # weak labels

    if cfg.TRAIN.OVERSAMPLE_SO:
        if cfg.MODEL.WEAK_LABELS:
            rois_obj, pos_vecs_obj, all_labels_obj, all_labels_obj_w, \
            neg_affinity_mask_obj, pos_affinity_mask_obj, \
            low_shot_ends_obj, regular_starts_obj = \
                _sample_rois_pos_neg_for_one_branch(
                    unique_all_rois_obj, unique_obj_gt_boxes,
                    unique_obj_gt_labels, unique_obj_gt_vecs,
                    low_shot_helper, 'obj', unique_obj_gt_labels_w)
        else:
            rois_obj, pos_vecs_obj, all_labels_obj, \
            neg_affinity_mask_obj, pos_affinity_mask_obj, \
            low_shot_ends_obj, regular_starts_obj = \
                _sample_rois_pos_neg_for_one_branch(
                    unique_all_rois_obj, unique_obj_gt_boxes,
                    unique_obj_gt_labels, unique_obj_gt_vecs,
                    low_shot_helper, 'obj')
    else:
        if cfg.MODEL.WEAK_LABELS:
            rois_obj, pos_vecs_obj, all_labels_obj, all_labels_obj_w, \
            neg_affinity_mask_obj, pos_affinity_mask_obj = \
                _sample_rois_pos_neg_for_one_branch(
                    unique_all_rois_obj, unique_obj_gt_boxes,
                    unique_obj_gt_labels, unique_obj_gt_vecs,
                    low_shot_helper, 'obj', unique_obj_gt_labels_w)
        else:
            rois_obj, pos_vecs_obj, all_labels_obj, \
            neg_affinity_mask_obj, pos_affinity_mask_obj = \
                _sample_rois_pos_neg_for_one_branch(
                    unique_all_rois_obj, unique_obj_gt_boxes,
                    unique_obj_gt_labels, unique_obj_gt_vecs,
                    low_shot_helper, 'obj')

    fg_size_obj = pos_vecs_obj.shape[0]
    pos_starts_obj = np.array([0, 0], dtype=np.int32)
    pos_ends_obj = np.array([fg_size_obj, -1], dtype=np.int32)
    neg_starts_obj = np.array([fg_size_obj, 0], dtype=np.int32)
    neg_ends_obj = np.array([-1, -1], dtype=np.int32)
    obj_pos_labels = all_labels_obj[:fg_size_obj] - 1
    if cfg.MODEL.WEAK_LABELS:
        obj_pos_labels_w = all_labels_obj_w[:fg_size_obj] - 1  # weak labels

    # Now sample rel rois
    rois_per_image = int(cfg.TRAIN.BATCH_SIZE_PER_IM)
    fg_rois_per_image = int(
        np.round(cfg.TRAIN.FG_FRACTION * rois_per_image))

    if cfg.TRAIN.OVERSAMPLE_SO:
        real_labels_sbj = all_labels_sbj[low_shot_ends_sbj[0]:]
        fg_inds_sbj = np.where(real_labels_sbj != 0)[0]
        bg_inds_sbj = np.where(real_labels_sbj == 0)[0]
    else:
        fg_inds_sbj = np.where(all_labels_sbj != 0)[0]
        bg_inds_sbj = np.where(all_labels_sbj == 0)[0]
    # # Only consider positive sbj rois for rel
    # # because during testing we assume each sbj roi is positive
    if cfg.MODEL.FOCAL_LOSS:
        num_fg_sbj, num_bg_sbj = len(fg_inds_sbj), len(bg_inds_sbj)
        out_num_fg_sbj = np.array([num_fg_sbj + 1.0], dtype=np.float32)
        out_num_bg_sbj = (np.array([num_bg_sbj + 1.0]) * (cfg.MODEL.NUM_CLASSES_SBJ_OBJ - 1) + out_num_fg_sbj * (
                cfg.MODEL.NUM_CLASSES_SBJ_OBJ - 2))
    # # we want the background amount to be equal to
    # # 0.125 * fg_rois_per_image if not smaller
    rel_bg_size_sbj = min(bg_inds_sbj.size, fg_rois_per_image)
    if rel_bg_size_sbj < bg_inds_sbj.size:
        rel_bg_inds_sbj = \
            npr.choice(bg_inds_sbj, size=rel_bg_size_sbj, replace=False)
    rel_keep_inds_sbj = np.append(fg_inds_sbj, rel_bg_inds_sbj)
    if cfg.TRAIN.OVERSAMPLE_SO:
        real_rois_sbj = rois_sbj[low_shot_ends_sbj[0]:]
        unique_rel_roi_sbj = real_rois_sbj[rel_keep_inds_sbj]
    else:
        unique_rel_roi_sbj = rois_sbj[rel_keep_inds_sbj]

    if cfg.TRAIN.OVERSAMPLE_SO:
        real_labels_obj = all_labels_obj[low_shot_ends_obj[0]:]
        fg_inds_obj = np.where(real_labels_obj != 0)[0]
        bg_inds_obj = np.where(real_labels_obj == 0)[0]
    else:
        fg_inds_obj = np.where(all_labels_obj != 0)[0]
        bg_inds_obj = np.where(all_labels_obj == 0)[0]
    # # Only consider positive obj rois for rel
    # # because during testing we assume each obj roi is positive
    if cfg.MODEL.FOCAL_LOSS:
        num_fg_obj, num_bg_obj = len(fg_inds_obj), len(bg_inds_obj)
        out_num_fg_obj = np.array([num_fg_obj + 1.0], dtype=np.float32)
        out_num_bg_obj = (np.array([num_bg_obj + 1.0]) * (cfg.MODEL.NUM_CLASSES_SBJ_OBJ - 1) + out_num_fg_obj * (
                cfg.MODEL.NUM_CLASSES_SBJ_OBJ - 2))
    # # we want the background amount to be equal to
    # # 0.125 * fg_rois_per_image if not smaller
    rel_bg_size_obj = min(bg_inds_obj.size, fg_rois_per_image)
    if rel_bg_size_obj < bg_inds_obj.size:
        rel_bg_inds_obj = \
            npr.choice(bg_inds_obj, size=rel_bg_size_obj, replace=False)
    rel_keep_inds_obj = np.append(fg_inds_obj, rel_bg_inds_obj)
    if cfg.TRAIN.OVERSAMPLE_SO:
        real_rois_obj = rois_obj[low_shot_ends_obj[0]:]
        unique_rel_roi_obj = real_rois_obj[rel_keep_inds_obj]
    else:
        unique_rel_roi_obj = rois_obj[rel_keep_inds_obj]

    # create potential relationships by considering all pairs
    rel_all_rois_sbj = np.repeat(unique_rel_roi_sbj, len(unique_rel_roi_obj), axis=0)
    rel_all_rois_obj = np.tile(unique_rel_roi_obj, (len(unique_rel_roi_sbj), 1))
    rel_all_rois_prd = box_union(rel_all_rois_sbj, rel_all_rois_obj)

    rel_overlaps_sbj = box_utils.bbox_overlaps(
        rel_all_rois_sbj[:, 1:5].astype(dtype=np.float32, copy=False),
        sbj_gt_boxes[:, :4].astype(dtype=np.float32, copy=False))

    rel_overlaps_obj = box_utils.bbox_overlaps(
        rel_all_rois_obj[:, 1:5].astype(dtype=np.float32, copy=False),
        obj_gt_boxes[:, :4].astype(dtype=np.float32, copy=False))

    # sample foreground candidates
    overlaps_pair_min = np.minimum(rel_overlaps_sbj, rel_overlaps_obj)
    max_overlaps_pair_min = overlaps_pair_min.max(axis=1)
    gt_assignment_pair_min = overlaps_pair_min.argmax(axis=1)
    rel_gt_inds = np.where((max_overlaps_pair_min >= 0.99999))[0]
    rel_pos_inds = np.where((max_overlaps_pair_min >= cfg.TRAIN.FG_THRESH) &
                            (max_overlaps_pair_min < 0.99999))[0]
    rel_fg_rois_per_this_image = min(int(fg_rois_per_image),
                                     rel_gt_inds.size + rel_pos_inds.size)
    if rel_pos_inds.size > 0 and \
            rel_pos_inds.size > fg_rois_per_image - rel_gt_inds.size:
        rel_pos_inds = npr.choice(rel_pos_inds,
                                  size=(rel_fg_rois_per_this_image - rel_gt_inds.size),
                                  replace=False)

    rel_fg_inds = np.append(rel_pos_inds, rel_gt_inds)
    # duplicate low-shot predicates to increase their chances to be chosen
    if cfg.TRAIN.OVERSAMPLE2:
        rel_pos_labels = rel_gt_labels[gt_assignment_pair_min[rel_fg_inds]] - 1
        low_shot_inds = \
            np.array([rel_fg_inds[i] for i, p in enumerate(rel_pos_labels) if
                      low_shot_helper.check_low_shot_p([-1, p, -1])], dtype=np.int32)
        rel_fg_inds = np.append(low_shot_inds, rel_fg_inds)
    if rel_fg_inds.size > fg_rois_per_image:
        rel_fg_inds = npr.choice(rel_fg_inds, size=fg_rois_per_image, replace=False)

    rel_bg_inds = np.where((max_overlaps_pair_min < cfg.TRAIN.BG_THRESH_HI))[0]
    rel_bg_rois_per_this_image = min(rois_per_image - rel_fg_inds.size,
                                     rois_per_image - fg_rois_per_image,
                                     rel_bg_inds.size)
    if rel_bg_inds.size > 0:
        rel_bg_inds = npr.choice(rel_bg_inds,
                                 size=rel_bg_rois_per_this_image,
                                 replace=False)
    if cfg.MODEL.FOCAL_LOSS:
        num_fg_rel, num_bg_rel = len(rel_fg_inds), len(rel_bg_inds)
        out_num_fg_rel = np.array([num_fg_rel + 1.0], dtype=np.float32)
        out_num_bg_rel = (
                np.array([num_bg_rel + 1.0]) * (cfg.MODEL.NUM_CLASSES_PRD - 1) +
                out_num_fg_rel * (cfg.MODEL.NUM_CLASSES_PRD - 2))

    # This oversampling method has redundant computation on those
    # low-shot ROIs, but it's flexible in that those low-shot ROIs
    # can be fed into the oversampler immediately after ROI-pooling,
    # instead of as late as after fc7
    if cfg.TRAIN.OVERSAMPLE:
        # Only consider low-shot on P
        rel_pos_labels = rel_gt_labels[gt_assignment_pair_min[rel_fg_inds]] - 1
        # low_shot_inds contains one dummy ROI at the very beginning
        # This is to make sure that low_shot ROIs are never empty
        low_shot_inds = \
            np.array([rel_fg_inds[i] for i, p in enumerate(rel_pos_labels) if
                      low_shot_helper.check_low_shot_p([-1, p, -1])], dtype=np.int32)
        rel_fg_inds = np.append(low_shot_inds, rel_fg_inds)
    if cfg.TRAIN.ADD_LOSS_WEIGHTS:
        # low-shot on P
        rel_pos_labels = rel_gt_labels[gt_assignment_pair_min[rel_fg_inds]] - 1
        rel_pos_weights = np.ones_like(rel_pos_labels, dtype=np.float32)
        low_shot_idx = \
            np.array([i for i, p in enumerate(rel_pos_labels) if
                      low_shot_helper.check_low_shot_p([-1, p, -1])], dtype=np.int32)
        rel_pos_weights[low_shot_idx] *= 2.0
        rel_pos_weights /= np.mean(rel_pos_weights)
    if cfg.TRAIN.ADD_LOSS_WEIGHTS_SO:
        # low-shot on S
        sbj_pos_weights = np.ones_like(sbj_pos_labels, dtype=np.float32)
        low_shot_idx = \
            np.array([i for i, s in enumerate(sbj_pos_labels) if
                      low_shot_helper.check_low_shot_s([s, -1, -1])], dtype=np.int32)
        sbj_pos_weights[low_shot_idx] *= 2.0
        sbj_pos_weights /= np.mean(sbj_pos_weights)
        # low-shot on O
        obj_pos_weights = np.ones_like(obj_pos_labels, dtype=np.float32)
        low_shot_idx = \
            np.array([i for i, o in enumerate(obj_pos_labels) if
                      low_shot_helper.check_low_shot_o([-1, -1, o])], dtype=np.int32)
        obj_pos_weights[low_shot_idx] *= 2.0
        obj_pos_weights /= np.mean(obj_pos_weights)

    rel_keep_inds = np.append(rel_fg_inds, rel_bg_inds)

    rel_rois_sbj = rel_all_rois_sbj[rel_keep_inds]
    rel_rois_obj = rel_all_rois_obj[rel_keep_inds]
    rel_rois_prd = rel_all_rois_prd[rel_keep_inds]

    pos_vecs_rel = rel_gt_vecs[gt_assignment_pair_min[rel_fg_inds]]

    all_labels_rel = rel_gt_labels[gt_assignment_pair_min[rel_keep_inds]]
    if cfg.MODEL.WEAK_LABELS:
        all_labels_rel_w = rel_gt_labels_w[gt_assignment_pair_min[rel_keep_inds]]  # weak labels

    rel_pos_labels = all_labels_rel[:rel_fg_inds.size] - 1
    if cfg.MODEL.WEAK_LABELS:
        rel_pos_labels_w = all_labels_rel_w[:rel_fg_inds.size] - 1  # weak labels

    all_labels_rel_horizontal_tile = np.tile(
        all_labels_rel, (rel_fg_inds.size, 1))
    all_labels_rel_vertical_tile = np.tile(
        all_labels_rel[:rel_fg_inds.size], (rel_keep_inds.size, 1)).transpose()
    neg_affinity_mask_rel = \
        np.array(all_labels_rel_horizontal_tile !=
                 all_labels_rel_vertical_tile).astype(np.float32)

    pos_starts_rel = np.array([0, 0], dtype=np.int32)
    pos_ends_rel = np.array([rel_fg_inds.size, -1], dtype=np.int32)
    neg_starts_rel = np.array([rel_fg_inds.size, 0], dtype=np.int32)
    neg_ends_rel = np.array([-1, -1], dtype=np.int32)

    blob = dict(
        sbj_rois=rois_sbj,
        obj_rois=rois_obj,
        rel_rois_sbj=rel_rois_sbj,
        rel_rois_obj=rel_rois_obj,
        rel_rois_prd=rel_rois_prd,
        sbj_pos_vecs=pos_vecs_sbj,
        obj_pos_vecs=pos_vecs_obj,
        rel_pos_vecs=pos_vecs_rel,
        sbj_pos_labels_int32=sbj_pos_labels.astype(np.int32, copy=False),
        obj_pos_labels_int32=obj_pos_labels.astype(np.int32, copy=False),
        rel_pos_labels_int32=rel_pos_labels.astype(np.int32, copy=False),
        sbj_neg_affinity_mask=neg_affinity_mask_sbj,
        obj_neg_affinity_mask=neg_affinity_mask_obj,
        rel_neg_affinity_mask=neg_affinity_mask_rel,
        sbj_pos_starts=pos_starts_sbj,
        obj_pos_starts=pos_starts_obj,
        rel_pos_starts=pos_starts_rel,
        sbj_pos_ends=pos_ends_sbj,
        obj_pos_ends=pos_ends_obj,
        rel_pos_ends=pos_ends_rel,
        sbj_neg_starts=neg_starts_sbj,
        obj_neg_starts=neg_starts_obj,
        rel_neg_starts=neg_starts_rel,
        sbj_neg_ends=neg_ends_sbj,
        obj_neg_ends=neg_ends_obj,
        rel_neg_ends=neg_ends_rel)

    if cfg.MODEL.FOCAL_LOSS:
        blob['fg_num_sbj'] = out_num_fg_sbj.astype(np.float32)
        blob['bg_num_sbj'] = out_num_bg_sbj.astype(np.float32)
        blob['fg_num_obj'] = out_num_fg_obj.astype(np.float32)
        blob['bg_num_obj'] = out_num_bg_obj.astype(np.float32)
        blob['fg_num_rel'] = out_num_fg_rel.astype(np.float32)
        blob['bg_num_rel'] = out_num_bg_rel.astype(np.float32)

    if cfg.MODEL.WEAK_LABELS:
        for num_w in range(cfg.MODEL.NUM_WEAK_LABELS):
            blob['sbj_pos_labels_int32_w_' + str(num_w)] = sbj_pos_labels_w[:, num_w].astype(np.int32,
                                                                                                  copy=False)  # weak labels
            blob['obj_pos_labels_int32_w_' + str(num_w)] = obj_pos_labels_w[:, num_w].astype(np.int32,
                                                                                                  copy=False)  # weak labels
            blob['rel_pos_labels_int32_w_' + str(num_w)] = rel_pos_labels_w[:, num_w].astype(np.int32,
                                                                                                  copy=False)  # weak labels

        blob['sbj_pos_labels_float32_w'] = np.zeros((sbj_pos_labels_w.shape[0], cfg.MODEL.NUM_CLASSES_SBJ_OBJ),
                                                    dtype=np.float32)
        blob['obj_pos_labels_float32_w'] = np.zeros((obj_pos_labels_w.shape[0], cfg.MODEL.NUM_CLASSES_SBJ_OBJ),
                                                    dtype=np.float32)
        blob['rel_pos_labels_float32_w'] = np.zeros((rel_pos_labels_w.shape[0], cfg.MODEL.NUM_CLASSES_PRD),
                                                    dtype=np.float32)
	
        overlap_sbj = np.array([(sbj_pos_labels[i] in sbj_pos_labels_w[i, :]) for i in range(len(sbj_pos_labels))])
        #overlap_sbj = np.isin(sbj_pos_labels, sbj_pos_labels_w)
        overlap_obj = np.array([(obj_pos_labels[i] in obj_pos_labels_w[i, :]) for i in range(len(obj_pos_labels))])
        #overlap_obj = np.isin(obj_pos_labels, obj_pos_labels_w)
        overlap_rel = np.array([(rel_pos_labels[i] in rel_pos_labels_w[i, :]) for i in range(len(rel_pos_labels))])
        #overlap_rel = np.isin(rel_pos_labels, rel_pos_labels_w)
        denominator_sbj = np.zeros((sbj_pos_labels_w.shape[0],), dtype=np.float32)
        denominator_obj = np.zeros((obj_pos_labels_w.shape[0],), dtype=np.float32)
        denominator_rel = np.zeros((rel_pos_labels_w.shape[0],), dtype=np.float32)

        denominator_sbj[np.where(overlap_sbj == False)] = 1.0 / (cfg.MODEL.NUM_WEAK_LABELS + 1)
        denominator_sbj[np.where(overlap_sbj == True)] = 1.0 / cfg.MODEL.NUM_WEAK_LABELS

        denominator_obj[np.where(overlap_obj == False)] = 1.0 / (cfg.MODEL.NUM_WEAK_LABELS + 1)
        denominator_obj[np.where(overlap_obj == True)] = 1.0 / cfg.MODEL.NUM_WEAK_LABELS

        denominator_rel[np.where(overlap_rel == False)] = 1.0 / (cfg.MODEL.NUM_WEAK_LABELS + 1)
        denominator_rel[np.where(overlap_rel == True)] = 1.0 / cfg.MODEL.NUM_WEAK_LABELS

        # while True:
        #	print('sbj_pos_labels_w', sbj_pos_labels_w)

        idx_batch_sbj = np.arange(sbj_pos_labels_w.shape[0], dtype=np.int32)
        idx_batch_obj = np.arange(obj_pos_labels_w.shape[0], dtype=np.int32)
        idx_batch_rel = np.arange(rel_pos_labels_w.shape[0], dtype=np.int32)
        # idx = np.stack([idx_batch, sbj_pos_labels_w[:, 1].astype(np.int32)], axis=1)

        # while True:
        # print('idx', idx.shape)
        # print("blob['sbj_pos_labels_float32_w']", blob['sbj_pos_labels_float32_w'][idx_batch, sbj_pos_labels_w[:, 1].astype(np.int32)])
        # print("blob['sbj_pos_labels_float32_w'][idx]", blob['sbj_pos_labels_float32_w'][idx])

        for num_w in range(cfg.MODEL.NUM_WEAK_LABELS):
            idx_sbj = sbj_pos_labels_w[:, num_w].astype(np.int32)
            idx_obj = obj_pos_labels_w[:, num_w].astype(np.int32)
            idx_rel = rel_pos_labels_w[:, num_w].astype(np.int32)

            blob['sbj_pos_labels_float32_w'][idx_batch_sbj, idx_sbj] = denominator_sbj
            blob['obj_pos_labels_float32_w'][idx_batch_obj, idx_obj] = denominator_obj
            blob['rel_pos_labels_float32_w'][idx_batch_rel, idx_rel] = denominator_rel

        idx_sbj = sbj_pos_labels.astype(np.int32)
        idx_obj = obj_pos_labels.astype(np.int32)
        idx_rel = rel_pos_labels.astype(np.int32)

        blob['sbj_pos_labels_float32_w'][idx_batch_sbj, idx_sbj] = denominator_sbj
        blob['obj_pos_labels_float32_w'][idx_batch_obj, idx_obj] = denominator_obj
        blob['rel_pos_labels_float32_w'][idx_batch_rel, idx_rel] = denominator_rel

        #while True:
            # print('idx_sbj', idx_sbj.shape)
            # print('idx_sbj', idx_sbj)
            # print('overlap_sbj', overlap_sbj.shape)
            # print('overlap_sbj', overlap_sbj)
            # print('denominator_sbj', denominator_sbj.shape)
            # print('denominator_sbj', denominator_sbj)
            #print('sbj_pos_labels.shape', sbj_pos_labels.shape)
            #print('sbj_pos_labels_w.shape', sbj_pos_labels_w.shape)
            #print('np.isin(sbj_pos_labels, sbj_pos_labels_w)', np.isin(sbj_pos_labels, sbj_pos_labels_w)[:5])
            #print('sbj_pos_labels', sbj_pos_labels[:5])
            #print('sbj_pos_labels_w', sbj_pos_labels_w[:5, :])
            #print('overlap_sbj', overlap_sbj[:5])
            #print('denominator_sbj', denominator_sbj[:5])
            #print('sbj_pos_labels_w[np.where(overlap_rel == True)]', sbj_pos_labels_w[np.where(overlap_rel == True)])
            #print('sbj_pos_labels[np.where(overlap_rel == True)]', sbj_pos_labels[np.where(overlap_rel == True)])
            # print("blob['sbj_pos_labels_float32_w']", blob['sbj_pos_labels_float32_w'].shape)
            #print("blob['sbj_pos_labels_float32_w']", blob['sbj_pos_labels_float32_w'][np.where(overlap_rel == True), :])
            #print("blob['sbj_pos_labels_float32_w'].sum", np.sum(blob['sbj_pos_labels_float32_w'], axis=1))
            # print("blob['sbj_pos_labels_int32']", blob['sbj_pos_labels_int32'].shape)

    if cfg.TRAIN.ADD_LOSS_WEIGHTS:
        blob['rel_pos_weights'] = rel_pos_weights
    if cfg.TRAIN.ADD_LOSS_WEIGHTS_SO:
        blob['sbj_pos_weights'] = sbj_pos_weights
        blob['obj_pos_weights'] = obj_pos_weights

    return blob


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


def _sample_rois_softmax_yall(
        unique_all_rois_sbj, unique_all_rois_obj,
        unique_sbj_gt_boxes, unique_obj_gt_boxes,
        unique_sbj_gt_vecs, unique_obj_gt_vecs,
        unique_sbj_gt_labels, unique_obj_gt_labels,
        sbj_gt_boxes, obj_gt_boxes,
        sbj_gt_vecs, obj_gt_vecs,
        rel_gt_vecs, rel_gt_labels,
        low_shot_helper):
    rois_sbj, pos_vecs_sbj, all_labels_sbj, _, _ = \
        _sample_rois_pos_neg_for_one_branch(
            unique_all_rois_sbj, unique_sbj_gt_boxes,
            unique_sbj_gt_labels, unique_sbj_gt_vecs,
            low_shot_helper, 'sbj')
    fg_size_sbj = pos_vecs_sbj.shape[0]
    pos_starts_sbj = np.array([0, 0], dtype=np.int32)
    pos_ends_sbj = np.array([fg_size_sbj, -1], dtype=np.int32)
    neg_starts_sbj = np.array([fg_size_sbj, 0], dtype=np.int32)
    neg_ends_sbj = np.array([-1, -1], dtype=np.int32)
    sbj_pos_labels = all_labels_sbj[:fg_size_sbj] - 1

    rois_obj, pos_vecs_obj, all_labels_obj, _, _ = \
        _sample_rois_pos_neg_for_one_branch(
            unique_all_rois_obj, unique_obj_gt_boxes,
            unique_obj_gt_labels, unique_obj_gt_vecs,
            low_shot_helper, 'obj')
    fg_size_obj = pos_vecs_obj.shape[0]
    pos_starts_obj = np.array([0, 0], dtype=np.int32)
    pos_ends_obj = np.array([fg_size_obj, -1], dtype=np.int32)
    neg_starts_obj = np.array([fg_size_obj, 0], dtype=np.int32)
    neg_ends_obj = np.array([-1, -1], dtype=np.int32)
    obj_pos_labels = all_labels_obj[:fg_size_obj] - 1

    rois_per_image = int(cfg.TRAIN.BATCH_SIZE_PER_IM)
    fg_rois_per_image = int(
        np.round(cfg.TRAIN.FG_FRACTION * rois_per_image))

    fg_inds_sbj = np.where(all_labels_sbj != 0)[0]
    # # Only consider positive sbj rois for rel
    # # because during testing we assume each sbj roi is positive
    bg_inds_sbj = np.where(all_labels_sbj == 0)[0]
    # # we want the background amount to be equal to
    # # 0.125 * fg_rois_per_image if not smaller
    rel_bg_size_sbj = min(bg_inds_sbj.size, fg_rois_per_image)
    if rel_bg_size_sbj < bg_inds_sbj.size:
        rel_bg_inds_sbj = \
            npr.choice(bg_inds_sbj, size=rel_bg_size_sbj, replace=False)
    rel_keep_inds_sbj = np.append(fg_inds_sbj, rel_bg_inds_sbj)
    unique_rel_roi_sbj = rois_sbj[rel_keep_inds_sbj]

    fg_inds_obj = np.where(all_labels_obj != 0)[0]
    # # Only consider positive obj rois for rel
    # # because during testing we assume each obj roi is positive
    bg_inds_obj = np.where(all_labels_obj == 0)[0]
    # # we want the background amount to be equal to
    # # 0.125 * fg_rois_per_image if not smaller
    rel_bg_size_obj = min(bg_inds_obj.size, fg_rois_per_image)
    if rel_bg_size_obj < bg_inds_obj.size:
        rel_bg_inds_obj = \
            npr.choice(bg_inds_obj, size=rel_bg_size_obj, replace=False)
    rel_keep_inds_obj = np.append(fg_inds_obj, rel_bg_inds_obj)
    unique_rel_roi_obj = rois_obj[rel_keep_inds_obj]

    # create potential relationships by considering all pairs
    rel_all_rois_sbj = np.repeat(unique_rel_roi_sbj, len(unique_rel_roi_obj), axis=0)
    rel_all_rois_obj = np.tile(unique_rel_roi_obj, (len(unique_rel_roi_sbj), 1))
    rel_all_rois_prd = box_union(rel_all_rois_sbj, rel_all_rois_obj)

    rel_overlaps_sbj = box_utils.bbox_overlaps(
        rel_all_rois_sbj[:, 1:5].astype(dtype=np.float32, copy=False),
        sbj_gt_boxes[:, :4].astype(dtype=np.float32, copy=False))

    rel_overlaps_obj = box_utils.bbox_overlaps(
        rel_all_rois_obj[:, 1:5].astype(dtype=np.float32, copy=False),
        obj_gt_boxes[:, :4].astype(dtype=np.float32, copy=False))

    # sample foreground candidates
    overlaps_pair_min = np.minimum(rel_overlaps_sbj, rel_overlaps_obj)
    max_overlaps_pair_min = overlaps_pair_min.max(axis=1)
    gt_assignment_pair_min = overlaps_pair_min.argmax(axis=1)
    rel_gt_inds = np.where((max_overlaps_pair_min >= 0.99999))[0]
    rel_pos_inds = np.where((max_overlaps_pair_min >= cfg.TRAIN.FG_THRESH) &
                            (max_overlaps_pair_min < 0.99999))[0]
    rel_fg_rois_per_this_image = min(int(fg_rois_per_image),
                                     rel_gt_inds.size + rel_pos_inds.size)
    if rel_pos_inds.size > 0 and \
            rel_pos_inds.size > fg_rois_per_image - rel_gt_inds.size:
        rel_pos_inds = npr.choice(rel_pos_inds,
                                  size=(rel_fg_rois_per_this_image - rel_gt_inds.size),
                                  replace=False)

    rel_fg_inds = np.append(rel_pos_inds, rel_gt_inds)
    if rel_fg_inds.size > fg_rois_per_image:
        rel_fg_inds = npr.choice(rel_fg_inds, size=fg_rois_per_image, replace=False)

    rel_bg_inds = np.where((max_overlaps_pair_min < cfg.TRAIN.BG_THRESH_HI))[0]
    rel_bg_rois_per_this_image = min(rois_per_image - rel_fg_inds.size,
                                     rois_per_image - fg_rois_per_image,
                                     rel_bg_inds.size)
    if rel_bg_inds.size > 0:
        rel_bg_inds = npr.choice(rel_bg_inds,
                                 size=rel_bg_rois_per_this_image,
                                 replace=False)

    rel_fg_inds = np.append(rel_pos_inds, rel_gt_inds)
    # duplicate low-shot predicates to increase their chances to be chosen
    if cfg.TRAIN.OVERSAMPLE2:
        rel_pos_labels = rel_gt_labels[gt_assignment_pair_min[rel_fg_inds]] - 1
        low_shot_inds = \
            np.array([rel_fg_inds[i] for i, p in enumerate(rel_pos_labels) if
                      low_shot_helper.check_low_shot_p([-1, p, -1])], dtype=np.int32)
        rel_fg_inds = np.append(low_shot_inds, rel_fg_inds)
    if rel_fg_inds.size > fg_rois_per_image:
        rel_fg_inds = npr.choice(rel_fg_inds, size=fg_rois_per_image, replace=False)

    rel_bg_inds = np.where((max_overlaps_pair_min < cfg.TRAIN.BG_THRESH_HI))[0]
    rel_bg_rois_per_this_image = min(rois_per_image - rel_fg_inds.size,
                                     rois_per_image - fg_rois_per_image,
                                     rel_bg_inds.size)
    if rel_bg_inds.size > 0:
        rel_bg_inds = npr.choice(rel_bg_inds,
                                 size=rel_bg_rois_per_this_image,
                                 replace=False)

    # This oversampling method has redundant computation on those
    # low-shot ROIs, but it's flexible in that those low-shot ROIs
    # can be fed into the oversampler immediately after ROI-pooling,
    # instead of as late as after fc7
    if cfg.TRAIN.OVERSAMPLE:
        # Only consider low-shot on P
        rel_pos_labels = rel_gt_labels[gt_assignment_pair_min[rel_fg_inds]] - 1
        # low_shot_inds contains one dummy ROI at the very beginning
        # This is to make sure that low_shot ROIs are never empty
        low_shot_inds = \
            np.array([rel_fg_inds[i] for i, p in enumerate(rel_pos_labels) if
                      low_shot_helper.check_low_shot_p([-1, p, -1])], dtype=np.int32)
        rel_fg_inds = np.append(low_shot_inds, rel_fg_inds)
    if cfg.TRAIN.ADD_LOSS_WEIGHTS:
        # low-shot on P
        rel_pos_labels = rel_gt_labels[gt_assignment_pair_min[rel_fg_inds]] - 1
        rel_pos_weights = np.ones_like(rel_pos_labels, dtype=np.float32)
        low_shot_idx = \
            np.array([i for i, p in enumerate(rel_pos_labels) if
                      low_shot_helper.check_low_shot_p([-1, p, -1])], dtype=np.int32)
        rel_pos_weights[low_shot_idx] *= 2.0
        rel_pos_weights /= np.mean(rel_pos_weights)
    if cfg.TRAIN.ADD_LOSS_WEIGHTS_SO:
        # low-shot on S
        sbj_pos_weights = np.ones_like(sbj_pos_labels, dtype=np.float32)
        low_shot_idx = \
            np.array([i for i, s in enumerate(sbj_pos_labels) if
                      low_shot_helper.check_low_shot_s([s, -1, -1])], dtype=np.int32)
        sbj_pos_weights[low_shot_idx] *= 2.0
        sbj_pos_weights /= np.mean(sbj_pos_weights)
        # low-shot on O
        obj_pos_weights = np.ones_like(obj_pos_labels, dtype=np.float32)
        low_shot_idx = \
            np.array([i for i, o in enumerate(obj_pos_labels) if
                      low_shot_helper.check_low_shot_o([-1, -1, o])], dtype=np.int32)
        obj_pos_weights[low_shot_idx] *= 2.0
        obj_pos_weights /= np.mean(obj_pos_weights)

    rel_keep_inds = np.append(rel_fg_inds, rel_bg_inds)

    rel_rois_sbj = rel_all_rois_sbj[rel_keep_inds]
    rel_rois_obj = rel_all_rois_obj[rel_keep_inds]
    rel_rois_prd = rel_all_rois_prd[rel_keep_inds]

    all_labels_rel = rel_gt_labels[gt_assignment_pair_min[rel_keep_inds]]
    rel_pos_labels = all_labels_rel[:rel_fg_inds.size] - 1

    pos_starts_rel = np.array([0, 0], dtype=np.int32)
    pos_ends_rel = np.array([rel_fg_inds.size, -1], dtype=np.int32)
    neg_starts_rel = np.array([rel_fg_inds.size, 0], dtype=np.int32)
    neg_ends_rel = np.array([-1, -1], dtype=np.int32)

    weight_sbj = np.zeros((cfg.MODEL.NUM_CLASSES_SBJ_OBJ, cfg.OUTPUT_EMBEDDING_DIM), dtype=np.float32)
    std = 1. / math.sqrt(weight_sbj.shape[1])
    weight_sbj = np.random.uniform(-std, std, (cfg.MODEL.NUM_CLASSES_SBJ_OBJ, cfg.OUTPUT_EMBEDDING_DIM)).astype(
        np.float32)

    weight_obj = weight_sbj

    weight_rel = np.zeros((cfg.MODEL.NUM_CLASSES_PRD, cfg.OUTPUT_EMBEDDING_DIM), dtype=np.float32)
    std = 1. / math.sqrt(weight_rel.shape[1])
    weight_rel = np.random.uniform(-std, std, (cfg.MODEL.NUM_CLASSES_PRD, cfg.OUTPUT_EMBEDDING_DIM)).astype(np.float32)

    # centroids_obj = load_pickle('centroids/centroids_obj.pkl')
    # centroids_rel = load_pickle('centroids/centroids_rel.pkl')

    # centroids_obj = centroids_obj.astype(np.float32)
    # centroids_rel = centroids_rel.astype(np.float32)

    blob = dict(
        sbj_rois=rois_sbj,
        obj_rois=rois_obj,
        rel_rois_sbj=rel_rois_sbj,
        rel_rois_obj=rel_rois_obj,
        rel_rois_prd=rel_rois_prd,
        sbj_pos_labels_int32=sbj_pos_labels.astype(np.int32, copy=False),
        obj_pos_labels_int32=obj_pos_labels.astype(np.int32, copy=False),
        rel_pos_labels_int32=rel_pos_labels.astype(np.int32, copy=False),
        sbj_pos_starts=pos_starts_sbj,
        obj_pos_starts=pos_starts_obj,
        rel_pos_starts=pos_starts_rel,
        sbj_pos_ends=pos_ends_sbj,
        obj_pos_ends=pos_ends_obj,
        rel_pos_ends=pos_ends_rel,
        sbj_neg_starts=neg_starts_sbj,
        obj_neg_starts=neg_starts_obj,
        rel_neg_starts=neg_starts_rel,
        sbj_neg_ends=neg_ends_sbj,
        obj_neg_ends=neg_ends_obj,
        rel_neg_ends=neg_ends_rel)
    if cfg.TRAIN.ADD_LOSS_WEIGHTS:
        blob['rel_pos_weights'] = rel_pos_weights
    if cfg.TRAIN.ADD_LOSS_WEIGHTS_SO:
        blob['sbj_pos_weights'] = sbj_pos_weights
        blob['obj_pos_weights'] = obj_pos_weights
    if cfg.MODEL.MEMORY_MODULE:
        blob['weight_sbj'] = weight_sbj
        blob['weight_obj'] = weight_obj
        blob['weight_rel'] = weight_rel
        # blob['centroids_sbj'] = centroids_obj
        # blob['centroids_obj'] = centroids_obj
        # blob['centroids_rel'] = centroids_rel

    return blob


def _sample_rois_pos_neg_for_one_branch(
        all_rois, gt_boxes, gt_labels, gt_vecs, low_shot_helper, label, gt_labels_w=None):
    rois_per_image = int(cfg.TRAIN.BATCH_SIZE_PER_IM)
    fg_rois_per_image = int(
        np.round(cfg.TRAIN.FG_FRACTION * rois_per_image))

    overlaps = box_utils.bbox_overlaps(
        all_rois[:, 1:5].astype(dtype=np.float32, copy=False),
        gt_boxes[:, :4].astype(dtype=np.float32, copy=False))
    max_overlaps = overlaps.max(axis=1)
    gt_assignment = overlaps.argmax(axis=1)

    gt_inds = np.where((max_overlaps >= 0.99999))[0]
    pos_inds = np.where((max_overlaps >= cfg.TRAIN.FG_THRESH) &
                        (max_overlaps < 0.99999))[0]
    fg_rois_per_this_image = min(int(fg_rois_per_image),
                                 gt_inds.size + pos_inds.size)
    if pos_inds.size > 0 and \
            pos_inds.size > fg_rois_per_image - gt_inds.size:
        pos_inds = npr.choice(pos_inds,
                              size=(fg_rois_per_this_image - gt_inds.size),
                              replace=False)
    fg_inds = np.append(pos_inds, gt_inds)
    # duplicate low-shot predicates to increase their chances to be chosen
    if cfg.TRAIN.OVERSAMPLE_SO2:
        pos_labels = gt_labels[gt_assignment[fg_inds]] - 1
        if label == 'sbj':
            low_shot_inds = \
                np.array([fg_inds[i] for i, s in enumerate(pos_labels) if
                          low_shot_helper.check_low_shot_s([s, -1, -1])], dtype=np.int32)
        elif label == 'obj':
            low_shot_inds = \
                np.array([fg_inds[i] for i, o in enumerate(pos_labels) if
                          low_shot_helper.check_low_shot_o([-1, -1, o])], dtype=np.int32)
        else:
            raise NotImplementedError
        fg_inds = np.append(low_shot_inds, fg_inds)
    if fg_inds.size > fg_rois_per_image:
        fg_inds = npr.choice(fg_inds, size=fg_rois_per_image, replace=False)

    bg_inds = np.where((max_overlaps < cfg.TRAIN.BG_THRESH_HI) &
                       (max_overlaps >= cfg.TRAIN.BG_THRESH_LO))[0]
    bg_rois_per_this_image = min(rois_per_image - fg_inds.size,
                                 rois_per_image - fg_rois_per_image,
                                 bg_inds.size)
    if bg_inds.size > 0:
        bg_inds = npr.choice(bg_inds, size=bg_rois_per_this_image, replace=False)

    if cfg.TRAIN.OVERSAMPLE_SO:
        pos_labels = gt_labels[gt_assignment[fg_inds]] - 1
        # low_shot_inds contains one dummy ROI at the very beginning
        # This is to make sure that low_shot ROIs are never empty
        if label == 'sbj':
            low_shot_inds = \
                np.array([fg_inds[i] for i, s in enumerate(pos_labels) if
                          low_shot_helper.check_low_shot_s([s, -1, -1])], dtype=np.int32)
        elif label == 'obj':
            low_shot_inds = \
                np.array([fg_inds[i] for i, o in enumerate(pos_labels) if
                          low_shot_helper.check_low_shot_o([-1, -1, o])], dtype=np.int32)
        else:
            raise NotImplementedError
        fg_inds = np.append(low_shot_inds, fg_inds)
        low_shot_ends = np.array([low_shot_inds.size, -1], dtype=np.int32)
        regular_starts = np.array([low_shot_inds.size, 0], dtype=np.int32)

    keep_inds = np.append(fg_inds, bg_inds)
    rois = all_rois[keep_inds]

    pos_vecs = gt_vecs[gt_assignment[fg_inds]]

    all_labels = np.zeros(len(keep_inds), dtype=np.float32)
    all_labels[:fg_inds.size] = gt_labels[gt_assignment[fg_inds]]

    if cfg.MODEL.WEAK_LABELS:
        all_labels_w = np.zeros((len(keep_inds), cfg.MODEL.NUM_WEAK_LABELS), dtype=np.float32)
        all_labels_w[:fg_inds.size] = gt_labels_w[gt_assignment[fg_inds]]

    all_labels_horizontal_tile = np.tile(
        all_labels, (fg_inds.size, 1))
    all_labels_vertical_tile = np.tile(
        all_labels[:fg_inds.size], (keep_inds.size, 1)).transpose()
    neg_affinity_mask = \
        np.array(all_labels_horizontal_tile !=
                 all_labels_vertical_tile).astype(np.float32)

    pos_labels_horizontal_tile = np.tile(
        all_labels[:fg_inds.size], (fg_inds.size, 1))
    pos_labels_vertical_tile = np.tile(
        all_labels[:fg_inds.size], (fg_inds.size, 1)).transpose()
    pos_affinity_mask = \
        np.array(pos_labels_horizontal_tile ==
                 pos_labels_vertical_tile).astype(np.float32)

    if cfg.TRAIN.OVERSAMPLE_SO:
        if cfg.MODEL.WEAK_LABELS:
            return rois, pos_vecs, all_labels, all_labels_w, neg_affinity_mask, pos_affinity_mask, low_shot_ends, regular_starts
        else:
            return rois, pos_vecs, all_labels, neg_affinity_mask, pos_affinity_mask, low_shot_ends, regular_starts

    else:
        if cfg.MODEL.WEAK_LABELS:
            return rois, pos_vecs, all_labels, all_labels_w, neg_affinity_mask, pos_affinity_mask
        else:
            return rois, pos_vecs, all_labels, neg_affinity_mask, pos_affinity_mask


def box_union(boxes1, boxes2):
    assert (boxes1[:, 0] == boxes2[:, 0]).all()
    xmin = np.minimum(boxes1[:, 1], boxes2[:, 1])
    ymin = np.minimum(boxes1[:, 2], boxes2[:, 2])
    xmax = np.maximum(boxes1[:, 3], boxes2[:, 3])
    ymax = np.maximum(boxes1[:, 4], boxes2[:, 4])
    return np.vstack((boxes1[:, 0], xmin, ymin, xmax, ymax)).transpose()
