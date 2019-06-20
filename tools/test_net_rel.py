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
import sys
import numpy as np
import math
import argparse
import pprint

import cv2  # NOQA (Must import before importing caffe2 due to bug in cv2)

from core.config_rel import (cfg, load_params_from_file, load_params_from_list)
from modeling import model_builder_rel
import utils.c2
import utils.train
from utils.timer import Timer
from utils.training_stats_rel import TrainingStats
import utils.env as envu
import utils.net_rel as nu
import utils.metrics_rel as metrics
from utils import helpers_rel
from utils import checkpoints_rel
from utils import evaluator_rel
import pickle
import json

from caffe2.python import workspace

import logging

FORMAT = '[%(levelname)s: %(filename)s: %(lineno)4d]: %(message)s'
logging.basicConfig(level=logging.INFO, format=FORMAT, stream=sys.stdout)
logger = logging.getLogger(__name__)

utils.c2.import_contrib_ops()
utils.c2.import_detectron_ops()

# OpenCL may be enabled by default in OpenCV3; disable it because it's not
# thread safe and causes unwanted GPU memory allocations.
cv2.ocl.setUseOpenCL(False)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Test a network'
    )
    parser.add_argument(
        '--cfg',
        dest='cfg_file',
        help='Config file for testing (and optionally testing)',
        default=None,
        type=str
    )
    parser.add_argument(
        'opts',
        help='See lib/core/config.py for all options',
        default=None,
        nargs=argparse.REMAINDER
    )
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    return parser.parse_args()


def test():
    test_model, output_dir, checkpoint_dir = \
        model_builder_rel.create(cfg.MODEL.MODEL_NAME, train=False, split='test')
    logger.info('Test model built.')
    total_test_iters = int(math.ceil(
        float(len(test_model.roi_data_loader._roidb)) / float(cfg.NUM_DEVICES))) + 5
    test_evaluator = evaluator_rel.Evaluator(
        split=cfg.TEST.DATA_TYPE,
        roidb_size=len(test_model.roi_data_loader._roidb))
    test_timer = Timer()
    logger.info('Test epoch iters: {}'.format(total_test_iters))

    accumulated_accs = {}
    for key in test_evaluator.__dict__.keys():
        if key.find('acc') >= 0:
            accumulated_accs[key] = []
    # wins are for showing different plots
    wins = {}
    for key in test_evaluator.__dict__.keys():
        if key.find('acc') >= 0:
            wins[key] = None

    params_file = cfg.TEST.WEIGHTS
    checkpoints_rel.initialize_params_from_file(
        model=test_model, weights_file=params_file,
        num_devices=cfg.NUM_DEVICES,
    )
    test_evaluator.reset()
    print("=> Testing model")
    for test_iter in range(0, total_test_iters):
        test_timer.tic()
        workspace.RunNet(test_model.net.Proto().name)
        test_timer.toc()
        test_evaluator.eval_im_dets_triplet_topk()
        logger.info('Tested {}/{} time: {}'.format(
            test_iter, total_test_iters, round(test_timer.average_time, 3)))
    iter_accs = test_evaluator.calculate_and_plot_accuracy()
    for key in iter_accs.keys():
        accumulated_accs[key].append(iter_accs[key])
    test_evaluator.save_all_dets()

    test_model.roi_data_loader.shutdown()

    logger.info('Testing has successfully finished...exiting!')

def test_enriched(split, save=False):
    if cfg.MODEL.WEAK_LABELS:
        detections_path = 'checkpoints/vg_wiki_and_relco/VGG16_reldn_fast_rcnn_conv4_spo_for_p/embd_fusion_w_relu_yall/8gpus_vgg16_softmaxed_triplet_no_last_l2norm_trainval_w_cluster_2_lan_layers_no_xpypxn/{}/reldn_detections_w.pkl'.format(
        split)
    else:
        detections_path = 'checkpoints/vg_wiki_and_relco/VGG16_reldn_fast_rcnn_conv4_spo_for_p/embd_fusion_w_relu_yall/8gpus_vgg16_softmaxed_triplet_no_last_l2norm_trainval_w_cluster_2_lan_layers_no_xpypxn/{}/reldn_detections.pkl'.format(
        split)

    detections = load_pickle(detections_path)
    total_test_iters = len(detections['image_idx'])
    test_evaluator = evaluator_rel.Evaluator(
        split=cfg.TEST.DATA_TYPE,
        roidb_size=total_test_iters)

    logger.info('Test epoch iters: {}'.format(total_test_iters))

    accumulated_accs = {}
    for key in test_evaluator.__dict__.keys():
        if key.find('acc') >= 0:
            accumulated_accs[key] = []
    # wins are for showing different plots
    wins = {}
    for key in test_evaluator.__dict__.keys():
        if key.find('acc') >= 0:
            wins[key] = None

    detections_results_keys = ['tri_top1', 'tri_top5', 'tri_top10',
                               'sbj_top1', 'sbj_top5', 'sbj_top10',
                               'obj_top1', 'obj_top5', 'obj_top10',
                               'rel_top1', 'rel_top5', 'rel_top10',
                               'tri_rr', 'sbj_rr', 'obj_rr', 'rel_rr',
                               'tri_mr', 'sbj_mr', 'obj_mr', 'rel_mr']

    detections_results = {key: [] for key in detections_results_keys}

    test_evaluator.reset()
    print("=> Evaluating results..")
    for test_iter in range(0, total_test_iters):
        image_idx = detections['image_idx'][test_iter]
        image_id = detections['image_id'][test_iter]
        gt_labels_sbj = detections['gt_labels_sbj'][test_iter]
        gt_labels_obj = detections['gt_labels_obj'][test_iter]
        gt_labels_rel = detections['gt_labels_rel'][test_iter]
        gt_boxes_sbj = detections['gt_boxes_sbj'][test_iter]
        gt_boxes_obj = detections['gt_boxes_obj'][test_iter]
        gt_boxes_rel = detections['gt_boxes_rel'][test_iter]
        det_boxes_sbj = detections['boxes_sbj'][test_iter]
        det_boxes_obj = detections['boxes_obj'][test_iter]
        det_boxes_rel = detections['boxes_rel'][test_iter]
        det_labels_sbj = detections['labels_sbj'][test_iter]
        det_labels_obj = detections['labels_obj'][test_iter]
        det_labels_rel = detections['labels_rel'][test_iter]
        det_scores_sbj = detections['scores_sbj'][test_iter]
        det_scores_obj = detections['scores_obj'][test_iter]
        det_scores_rel = detections['scores_rel'][test_iter]

        detections_results_image = test_evaluator.eval_im_dets_triplet_topk_enriched(image_idx,
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
                                                                                     det_scores_rel,
                                                                                     detections_results_keys)

        # print(detections_results_keys)
        for key in detections_results_keys:
            detections_results[key].append(detections_results_image[key])

    iter_accs = test_evaluator.calculate_and_plot_accuracy()

    for key in iter_accs.keys():
        accumulated_accs[key].append(iter_accs[key])

    if (save):
        with open('./accuracies/accuracies_{}.json'.format(split), 'w') as fp:
            json.dump(accumulated_accs, fp)
        with open('./accuracies/detections_results_{}.json'.format(split), 'w') as fp:
            json.dump(detections_results, fp)

    logger.info('Testing has successfully finished...exiting!')

    return accumulated_accs, detections_results


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



if __name__ == '__main__':

    args = parse_args()
    logger.info('Called with args:')
    logger.info(args)
    if args.cfg_file is not None:
        load_params_from_file(args.cfg_file)
    if args.opts is not None:
        load_params_from_list(args.opts)
    logger.info('Testing with config:')
    logger.info(pprint.pformat(cfg))

    # test()
    test_enriched(cfg.TEST.DATA_TYPE, save=True)