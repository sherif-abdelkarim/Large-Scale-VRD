# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
import numpy as np
import cPickle

from core.config_rel import cfg
from datasets.factory import get_imdb

import logging
logger = logging.getLogger(__name__)

DEBUG = False


def combined_roidb_for_val_test(dataset_names): # Comment changes until proven needed
    def get_roidb(dataset_name):

        logger.info('loading roidb for {}'.format(dataset_name))

        if cfg.MODEL.WEAK_LABELS:
            roidb_file = os.path.join(cfg.DATA_DIR, 'roidb_cache', dataset_name +
                                      '_configured_gt_roidb_w{}.pkl'.format(cfg.RNG_SEED))
        else:
            roidb_file = os.path.join(cfg.DATA_DIR, 'roidb_cache', dataset_name +
                                      '_configured_gt_roidb{}.pkl'.format(cfg.RNG_SEED))

        if os.path.exists(roidb_file):
            with open(roidb_file, 'rb') as fid:
                roidb = cPickle.load(fid)
            logger.info('len(roidb): {}'.format(len(roidb)))
            logger.info('{} configured gt roidb loaded from {}'.format(
                dataset_name, roidb_file))
            return roidb

        ds = get_imdb(dataset_name)
        roidb = ds.gt_roidb()
        logger.info('loading widths and appending them')
        widths, heights = ds.get_widths_and_heights()

        for i in range(len(roidb)):
            logger.info('creating roidb for image {}'.format(i + 1))
            roidb[i]['width'] = widths[i]
            roidb[i]['height'] = heights[i]
            roidb[i]['image'] = ds.image_path_at(i)
            gt_sbj_overlaps = roidb[i]['gt_sbj_overlaps'].toarray()
            if cfg.MODEL.WEAK_LABELS:
                gt_sbj_overlaps_w = roidb[i]['gt_sbj_overlaps_w'].todense()
            # max sbj_overlap with gt over classes (columns)
            sbj_max_overlaps = gt_sbj_overlaps.max(axis=1)
            if cfg.MODEL.WEAK_LABELS:
                sbj_max_overlaps_w = gt_sbj_overlaps_w.max(axis=2)

            # gt sbj_class that had the max sbj_overlap
            sbj_max_classes = gt_sbj_overlaps.argmax(axis=1)
            if cfg.MODEL.WEAK_LABELS:
                sbj_max_classes_w = gt_sbj_overlaps_w.argmax(axis=2)

            roidb[i]['sbj_max_classes'] = sbj_max_classes
            roidb[i]['sbj_max_overlaps'] = sbj_max_overlaps
            if cfg.MODEL.WEAK_LABELS:
                roidb[i]['sbj_max_classes_w'] = sbj_max_classes_w
                roidb[i]['sbj_max_overlaps_w'] = sbj_max_overlaps_w

            # sanity checks
            # max overlap of 0 => class should be zero (background)
            zero_inds = np.where(sbj_max_overlaps == 0)[0]
            assert all(sbj_max_classes[zero_inds] == 0)
            # max overlap > 0 => class should not be zero (must be a fg class)
            nonzero_inds = np.where(sbj_max_overlaps > 0)[0]
            assert all(sbj_max_classes[nonzero_inds] != 0)

            # need gt_obj_overlaps as a dense array for argmax
            gt_obj_overlaps = roidb[i]['gt_obj_overlaps'].toarray()
            if cfg.MODEL.WEAK_LABELS:
                gt_obj_overlaps_w = roidb[i]['gt_obj_overlaps_w'].todense()
            # max obj_overlap with gt over classes (columns)
            obj_max_overlaps = gt_obj_overlaps.max(axis=1)
            if cfg.MODEL.WEAK_LABELS:
                obj_max_overlaps_w = gt_obj_overlaps_w.max(axis=2)
            # gt obj_class that had the max obj_overlap
            obj_max_classes = gt_obj_overlaps.argmax(axis=1)
            if cfg.MODEL.WEAK_LABELS:
                obj_max_classes_w = gt_obj_overlaps_w.argmax(axis=2)

            roidb[i]['obj_max_classes'] = obj_max_classes
            roidb[i]['obj_max_overlaps'] = obj_max_overlaps
            if cfg.MODEL.WEAK_LABELS:
                roidb[i]['obj_max_classes_w'] = obj_max_classes_w
                roidb[i]['obj_max_overlaps_w'] = obj_max_overlaps_w

            # sanity checks
            # max overlap of 0 => class should be zero (background)
            zero_inds = np.where(obj_max_overlaps == 0)[0]
            assert all(obj_max_classes[zero_inds] == 0)
            # max overlap > 0 => class should not be zero (must be a fg class)
            nonzero_inds = np.where(obj_max_overlaps > 0)[0]
            assert all(obj_max_classes[nonzero_inds] != 0)

            # need gt_rel_overlaps as a dense array for argmax
            gt_rel_overlaps = roidb[i]['gt_rel_overlaps'].toarray()
            if cfg.MODEL.WEAK_LABELS:
                gt_rel_overlaps_w = roidb[i]['gt_rel_overlaps_w'].todense()
            # max rel_overlap with gt over classes (columns)
            rel_max_overlaps = gt_rel_overlaps.max(axis=1)
            if cfg.MODEL.WEAK_LABELS:
                rel_max_overlaps_w = gt_rel_overlaps_w.max(axis=2)
            # gt rel_class that had the max rel_overlap
            rel_max_classes = gt_rel_overlaps.argmax(axis=1)
            if cfg.MODEL.WEAK_LABELS:
                rel_max_classes_w = gt_rel_overlaps_w.argmax(axis=2)
            roidb[i]['rel_max_classes'] = rel_max_classes
            roidb[i]['rel_max_overlaps'] = rel_max_overlaps
            if cfg.MODEL.WEAK_LABELS:
                roidb[i]['rel_max_classes_w'] = rel_max_classes_w
                roidb[i]['rel_max_overlaps_w'] = rel_max_overlaps_w
            # sanity checks
            # max overlap of 0 => class should be zero (background)
            zero_inds = np.where(rel_max_overlaps == 0)[0]
            assert all(rel_max_classes[zero_inds] == 0)
            # max overlap > 0 => class should not be zero (must be a fg class)
            nonzero_inds = np.where(rel_max_overlaps > 0)[0]
            assert all(rel_max_classes[nonzero_inds] != 0)

#             zero_inds_w = np.where(rel_max_overlaps_w == 0)[0]
# #             assert all(rel_max_classes_w[zero_inds_w] == 0)
            # max overlap > 0 => class should not be zero (must be a fg class)
#             nonzero_inds_w = np.where(rel_max_overlaps_w > 0)[0]
# #             assert all(rel_max_classes_w[nonzero_inds_w] != 0)

        logger.info('Loaded dataset: {:s}'.format(ds.name))
        logger.info('len(roidb): {}'.format(len(roidb)))

        with open(roidb_file, 'wb') as fid:
            cPickle.dump(roidb, fid, cPickle.HIGHEST_PROTOCOL)
        logger.info('wrote configured gt roidb to {}'.format(roidb_file))

        return roidb

    dataset_names = dataset_names.split(':')
    roidbs = [get_roidb(*args) for args in zip(dataset_names)]
    roidb = roidbs[0]
    for r in roidbs[1:]:
        roidb.extend(r)

    if cfg.VAL.PROPOSAL_FILE != '':
        with open(cfg.VAL.PROPOSAL_FILE, 'rb') as fid:
            proposals = cPickle.load(fid)
        return roidb, proposals
    else:
        return roidb


def combined_roidb_for_training(dataset_names, proposal_files): # Comment changes until proven needed
    def get_roidb(dataset_name, proposal_file):

        logger.info('loading roidb for {}'.format(dataset_name))

        if cfg.MODEL.WEAK_LABELS:
            roidb_file = os.path.join(cfg.DATA_DIR, 'roidb_cache', dataset_name +
                                      '_configured_gt_roidb_w{}.pkl'.format(cfg.RNG_SEED))
        else:
            roidb_file = os.path.join(cfg.DATA_DIR, 'roidb_cache', dataset_name +
                                      '_configured_gt_roidb{}.pkl'.format(cfg.RNG_SEED))

        if os.path.exists(roidb_file):
            with open(roidb_file, 'rb') as fid:
                roidb = cPickle.load(fid)
            logger.info('len(roidb): {}'.format(len(roidb)))
            logger.info('{} configured gt roidb loaded from {}'.format(
                dataset_name, roidb_file))

            if cfg.TRAIN.USE_FLIPPED:
                logger.info('Appending horizontally-flipped training examples...')
                extend_with_flipped_entries(roidb)
            logger.info('Loaded dataset: {:s}'.format(dataset_name))

            return roidb

        ds = get_imdb(dataset_name)
        roidb = ds.gt_roidb()
        logger.info('loading widths and appending them')
        widths, heights = ds.get_widths_and_heights()

        for i in range(len(roidb)):
            logger.info('creating roidb for image {}'.format(i + 1))
            roidb[i]['width'] = widths[i]
            roidb[i]['height'] = heights[i]
            roidb[i]['image'] = ds.image_path_at(i)
            gt_sbj_overlaps = roidb[i]['gt_sbj_overlaps'].toarray()
            if cfg.MODEL.WEAK_LABELS:
                gt_sbj_overlaps_w = roidb[i]['gt_sbj_overlaps_w'].todense()
            # max sbj_overlap with gt over classes (columns)
            sbj_max_overlaps = gt_sbj_overlaps.max(axis=1)
            if cfg.MODEL.WEAK_LABELS:
                sbj_max_overlaps_w = gt_sbj_overlaps_w.max(axis=2)

            # gt sbj_class that had the max sbj_overlap
            sbj_max_classes = gt_sbj_overlaps.argmax(axis=1)
            if cfg.MODEL.WEAK_LABELS:
                sbj_max_classes_w = gt_sbj_overlaps_w.argmax(axis=2)

            roidb[i]['sbj_max_classes'] = sbj_max_classes
            roidb[i]['sbj_max_overlaps'] = sbj_max_overlaps
            if cfg.MODEL.WEAK_LABELS:
                roidb[i]['sbj_max_classes_w'] = sbj_max_classes_w
                roidb[i]['sbj_max_overlaps_w'] = sbj_max_overlaps_w

            # sanity checks
            # max overlap of 0 => class should be zero (background)
            zero_inds = np.where(sbj_max_overlaps == 0)[0]
            assert all(sbj_max_classes[zero_inds] == 0)
            # max overlap > 0 => class should not be zero (must be a fg class)
            nonzero_inds = np.where(sbj_max_overlaps > 0)[0]
            assert all(sbj_max_classes[nonzero_inds] != 0)

            # need gt_obj_overlaps as a dense array for argmax
            gt_obj_overlaps = roidb[i]['gt_obj_overlaps'].toarray()
            if cfg.MODEL.WEAK_LABELS:
                gt_obj_overlaps_w = roidb[i]['gt_obj_overlaps_w'].todense()
            # max obj_overlap with gt over classes (columns)
            obj_max_overlaps = gt_obj_overlaps.max(axis=1)
            if cfg.MODEL.WEAK_LABELS:
                obj_max_overlaps_w = gt_obj_overlaps_w.max(axis=2)
            # gt obj_class that had the max obj_overlap
            obj_max_classes = gt_obj_overlaps.argmax(axis=1)
            if cfg.MODEL.WEAK_LABELS:
                obj_max_classes_w = gt_obj_overlaps_w.argmax(axis=2)

            roidb[i]['obj_max_classes'] = obj_max_classes
            roidb[i]['obj_max_overlaps'] = obj_max_overlaps
            if cfg.MODEL.WEAK_LABELS:
                roidb[i]['obj_max_classes_w'] = obj_max_classes_w
                roidb[i]['obj_max_overlaps_w'] = obj_max_overlaps_w

            # sanity checks
            # max overlap of 0 => class should be zero (background)
            zero_inds = np.where(obj_max_overlaps == 0)[0]
            assert all(obj_max_classes[zero_inds] == 0)
            # max overlap > 0 => class should not be zero (must be a fg class)
            nonzero_inds = np.where(obj_max_overlaps > 0)[0]
            assert all(obj_max_classes[nonzero_inds] != 0)

            # need gt_rel_overlaps as a dense array for argmax
            gt_rel_overlaps = roidb[i]['gt_rel_overlaps'].toarray()
            if cfg.MODEL.WEAK_LABELS:
                gt_rel_overlaps_w = roidb[i]['gt_rel_overlaps_w'].todense()
            # max rel_overlap with gt over classes (columns)
            rel_max_overlaps = gt_rel_overlaps.max(axis=1)
            if cfg.MODEL.WEAK_LABELS:
                rel_max_overlaps_w = gt_rel_overlaps_w.max(axis=2)
            # gt rel_class that had the max rel_overlap
            rel_max_classes = gt_rel_overlaps.argmax(axis=1)
            if cfg.MODEL.WEAK_LABELS:
                rel_max_classes_w = gt_rel_overlaps_w.argmax(axis=2)
            roidb[i]['rel_max_classes'] = rel_max_classes
            roidb[i]['rel_max_overlaps'] = rel_max_overlaps
            if cfg.MODEL.WEAK_LABELS:
                roidb[i]['rel_max_classes_w'] = rel_max_classes_w
                roidb[i]['rel_max_overlaps_w'] = rel_max_overlaps_w
            # sanity checks
            # max overlap of 0 => class should be zero (background)
            zero_inds = np.where(rel_max_overlaps == 0)[0]
            assert all(rel_max_classes[zero_inds] == 0)
            # max overlap > 0 => class should not be zero (must be a fg class)
            nonzero_inds = np.where(rel_max_overlaps > 0)[0]
            assert all(rel_max_classes[nonzero_inds] != 0)

#             zero_inds_w = np.where(rel_max_overlaps_w == 0)[0]
# #             assert all(rel_max_classes_w[zero_inds_w] == 0)
            # max overlap > 0 => class should not be zero (must be a fg class)
#             nonzero_inds_w = np.where(rel_max_overlaps_w > 0)[0]
# #             assert all(rel_max_classes_w[nonzero_inds_w] != 0)

        logger.info('Loaded dataset: {:s}'.format(ds.name))
        logger.info('len(roidb): {}'.format(len(roidb)))

        with open(roidb_file, 'wb') as fid:
            cPickle.dump(roidb, fid, cPickle.HIGHEST_PROTOCOL)
        logger.info('wrote configured gt roidb to {}'.format(roidb_file))

        if cfg.TRAIN.USE_FLIPPED:
            logger.info('Appending horizontally-flipped training examples...')
            extend_with_flipped_entries(roidb)
        logger.info('Loaded dataset: {:s}'.format(dataset_name))

        return roidb

    dataset_names = dataset_names.split(':')
    if proposal_files is not None:
        proposal_files = proposal_files.split(':')
    else:
        proposal_files = [None] * len(dataset_names)
    assert len(dataset_names) == len(proposal_files)
    roidbs = [get_roidb(*args) for args in zip(dataset_names, proposal_files)]
    roidb = roidbs[0]
    for r in roidbs[1:]:
        roidb.extend(r)

    roidb = filter_for_training(roidb)

    if cfg.TRAIN.PROPOSAL_FILE != '':
        with open(cfg.TRAIN.PROPOSAL_FILE, 'rb') as fid:
            proposals = cPickle.load(fid)
        return roidb, proposals
    else:
        return roidb


def extend_with_flipped_entries(roidb):

    flipped_roidb = []
    for entry in roidb:
        width = entry['width']
        height = entry['height']
        entry['sbj_boxes'][:, 2] = np.minimum(entry['sbj_boxes'][:, 2], width - 1)
        entry['sbj_boxes'][:, 3] = np.minimum(entry['sbj_boxes'][:, 3], height - 1)
        entry['obj_boxes'][:, 2] = np.minimum(entry['obj_boxes'][:, 2], width - 1)
        entry['obj_boxes'][:, 3] = np.minimum(entry['obj_boxes'][:, 3], height - 1)
        entry['rel_boxes'][:, 2] = np.minimum(entry['rel_boxes'][:, 2], width - 1)
        entry['rel_boxes'][:, 3] = np.minimum(entry['rel_boxes'][:, 3], height - 1)

    for entry in roidb:
        width = entry['width']
        height = entry['height']

        sbj_boxes = entry['sbj_boxes'].copy()
        oldx1 = sbj_boxes[:, 0].copy()
        oldx2 = sbj_boxes[:, 2].copy()
        sbj_boxes[:, 0] = width - oldx2 - 1
        sbj_boxes[:, 2] = width - oldx1 - 1
        assert (sbj_boxes[:, 2] >= sbj_boxes[:, 0]).all()

        obj_boxes = entry['obj_boxes'].copy()
        oldx1 = obj_boxes[:, 0].copy()
        oldx2 = obj_boxes[:, 2].copy()
        obj_boxes[:, 0] = width - oldx2 - 1
        obj_boxes[:, 2] = width - oldx1 - 1
        assert (obj_boxes[:, 2] >= obj_boxes[:, 0]).all()

        rel_boxes = entry['rel_boxes'].copy()
        oldx1 = rel_boxes[:, 0].copy()
        oldx2 = rel_boxes[:, 2].copy()
        rel_boxes[:, 0] = width - oldx2 - 1
        rel_boxes[:, 2] = width - oldx1 - 1
        assert (rel_boxes[:, 2] >= rel_boxes[:, 0]).all()

        flipped_entry = {}
        dont_copy = \
            ('obj_boxes', 'sbj_boxes', 'rel_boxes', 'segms', 'gt_keypoints', 'flipped')
        for k, v in entry.items():
            if k not in dont_copy:
                flipped_entry[k] = v
        flipped_entry['sbj_boxes'] = sbj_boxes
        flipped_entry['obj_boxes'] = obj_boxes
        flipped_entry['rel_boxes'] = rel_boxes

        flipped_entry['flipped'] = True
        flipped_roidb.append(flipped_entry)
    roidb.extend(flipped_roidb)


def filter_for_training(roidb):

    """Remove roidb entries that have no usable RoIs based on config settings.
    """
    def is_valid(entry):
        # Valid images have:
        #   (1) At least one foreground RoI OR
        #   (2) At least one background RoI
        sbj_overlaps = entry['sbj_max_overlaps']
        # find boxes with sufficient overlap
        sbj_fg_inds = np.where(sbj_overlaps >= cfg.TRAIN.FG_THRESH)[0]
        # Select background RoIs as those within [BG_THRESH_LO, BG_THRESH_HI)
        sbj_bg_inds = np.where((sbj_overlaps < cfg.TRAIN.BG_THRESH_HI) &
                               (sbj_overlaps >= cfg.TRAIN.BG_THRESH_LO))[0]
        # image is only valid if such boxes exist
        sbj_valid = len(sbj_fg_inds) > 0 or len(sbj_bg_inds) > 0

        obj_overlaps = entry['obj_max_overlaps']
        # find boxes with sufficient overlap
        obj_fg_inds = np.where(obj_overlaps >= cfg.TRAIN.FG_THRESH)[0]
        # Select background RoIs as those within [BG_THRESH_LO, BG_THRESH_HI)
        obj_bg_inds = np.where((obj_overlaps < cfg.TRAIN.BG_THRESH_HI) &
                               (obj_overlaps >= cfg.TRAIN.BG_THRESH_LO))[0]
        # image is only valid if such boxes exist
        obj_valid = len(obj_fg_inds) > 0 or len(obj_bg_inds) > 0

        rel_overlaps = entry['rel_max_overlaps']
        # find boxes with sufficient overlap
        rel_fg_inds = np.where(rel_overlaps >= cfg.TRAIN.FG_THRESH)[0]
        # Select background RoIs as those within [BG_THRESH_LO, BG_THRESH_HI)
        rel_bg_inds = np.where((rel_overlaps < cfg.TRAIN.BG_THRESH_HI) &
                               (rel_overlaps >= cfg.TRAIN.BG_THRESH_LO))[0]
        # image is only valid if such boxes exist
        rel_valid = len(rel_fg_inds) > 0 or len(rel_bg_inds) > 0

        valid = sbj_valid and obj_valid and rel_valid

        return valid

    num = len(roidb)
    filtered_roidb = [entry for entry in roidb if is_valid(entry)]
    num_after = len(filtered_roidb)
    logger.info('Filtered {} roidb entries: {} -> {}'.
                format(num - num_after, num, num_after))
    return filtered_roidb
