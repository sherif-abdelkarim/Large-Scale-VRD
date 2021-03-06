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
import json

import cv2  # NOQA (Must import before importing caffe2 due to bug in cv2)

from core.config_rel import (cfg, load_params_from_file, load_params_from_list)
from modeling import model_builder_rel
import utils.c2
from utils.timer import Timer
from utils.training_stats_rel import TrainingStats
import utils.env as envu
import utils.net_rel as nu
import utils.metrics_rel as metrics
from utils import helpers_rel
from utils import checkpoints_rel
from utils import evaluator_rel

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
        description='Train a network with Detectron'
    )
    parser.add_argument(
        '--cfg',
        dest='cfg_file',
        help='Config file for training (and optionally testing)',
        default=None,
        type=str
    )
    parser.add_argument(
        '--multi-gpu-testing',
        dest='multi_gpu_testing',
        help='Use cfg.NUM_GPUS GPUs for inference',
        action='store_true'
    )
    parser.add_argument(
        '--skip-test',
        dest='skip_test',
        help='Do not test the final model',
        action='store_true'
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


if __name__ == '__main__':

    args = parse_args()
    logger.info('Called with args:')
    logger.info(args)
    if args.cfg_file is not None:
        load_params_from_file(args.cfg_file)
    if args.opts is not None:
        load_params_from_list(args.opts)

    helpers_rel.set_random_seed(cfg.RNG_SEED)
    logger.info('Training with config:')
    logger.info(pprint.pformat(cfg))

    train_model, output_dir, checkpoint_dir = \
        model_builder_rel.create(cfg.MODEL.MODEL_NAME, train=True)
    logger.info('Training model built.')
    start_model_iter = 0
    params_ckp_file = checkpoints_rel.get_checkpoint_resume_file(checkpoint_dir)
    logger.info('params_ckp_file = {}, CHECKPOINT.CHECKPOINT_MODEL ={}'.format(params_ckp_file, cfg.CHECKPOINT.CHECKPOINT_MODEL))
    if (cfg.CHECKPOINT.CHECKPOINT_MODEL and params_ckp_file is not None):
        params_ckp_file = checkpoints_rel.get_checkpoint_resume_file(checkpoint_dir)
        start_model_iter = int(os.path.basename(params_ckp_file).replace('.pkl', '').replace('c2_model_iter', ''))
        checkpoints_rel.initialize_params_from_file(model=train_model,
                                                    weights_file=params_ckp_file,
                                                    num_devices=cfg.NUM_DEVICES,
                                                    )

    # do validation by default
    if True:
        val_model, _, _ = \
            model_builder_rel.create(cfg.MODEL.MODEL_NAME, train=False, split='val')
        logger.info('Validation model built.')
        if (cfg.CHECKPOINT.CHECKPOINT_MODEL and params_ckp_file is not None):
            params_ckp_file = checkpoints_rel.get_checkpoint_resume_file(checkpoint_dir)
            checkpoints_rel.initialize_params_from_file(model=val_model,
                                                        weights_file=params_ckp_file,
                                                        num_devices=cfg.NUM_DEVICES,
                                                        )
        total_val_iters = int(math.ceil(
            float(len(val_model.roi_data_loader._roidb)) / float(cfg.NUM_DEVICES))) + 5
        val_evaluator = evaluator_rel.Evaluator(
            split=cfg.VAL.DATA_TYPE,
            roidb_size=len(val_model.roi_data_loader._roidb))
        val_timer = Timer()
        logger.info('Val epoch iters: {}'.format(total_val_iters))

        accumulated_accs = {}
        for key in val_evaluator.__dict__.keys():
            if key.find('acc') >= 0:
                accumulated_accs[key] = []
        # wins are for showing different plots
        wins = {}
        for key in val_evaluator.__dict__.keys():
            if key.find('acc') >= 0:
                wins[key] = None

    prev_checkpointed_lr = None

    lr_iters = model_builder_rel.get_lr_steps()
    train_timer = Timer()

    device_prefix = 'gpu_' if cfg.DEVICE == 'GPU' else 'cpu_'
    prefix = '{}{}'.format(device_prefix, cfg.ROOT_DEVICE_ID)
    train_metrics_calculator = metrics.MetricsCalculator(
        model=train_model,
        split='train',
        batch_size=cfg.TRAIN.IMS_PER_BATCH / cfg.MODEL.GRAD_ACCUM_FREQUENCY,
        prefix=prefix
    )
    checkpoints = {}

    if os.path.exists(os.path.join(checkpoint_dir, 'best.json')):
        best = json.load(open(os.path.join(checkpoint_dir, 'best.json')))
    else:
        best = {}
        best['best_avg_top1_acc'] = 0.0
        best['iteration'] = 0
        best['accuracies'] = []
        json.dump(best, open(os.path.join(checkpoint_dir, 'best.json'), 'w'))


    for curr_iter in range(start_model_iter, cfg.SOLVER.NUM_ITERATIONS):

        lr = model_builder_rel.add_variable_stepsize_lr(
            curr_iter + 1, cfg.NUM_DEVICES, lr_iters, start_model_iter + 1,
            train_model, prev_checkpointed_lr
        )
        train_timer.tic()
        workspace.RunNet(train_model.net.Proto().name)
        train_timer.toc(average=False)

        mb_size = train_model.roi_data_loader.minibatch_queue_size()
        if curr_iter == start_model_iter:
            helpers_rel.print_net(train_model)
            if cfg.DEVICE == 'GPU':
                os.system('nvidia-smi')
            model_flops, model_params = helpers_rel.get_flops_params(
                train_model, cfg.ROOT_DEVICE_ID)
            print('Total network FLOPs (10^9): {}'.format(model_flops))
            print('Total network params (10^6): {}'.format(model_params))
        helpers_rel.check_nan_losses(train_model, cfg.NUM_DEVICES)

        rem_train_iters = (
                (cfg.SOLVER.NUM_ITERATIONS - curr_iter - 1) *
                cfg.MODEL.GRAD_ACCUM_FREQUENCY
        )
        train_metrics_calculator.calculate_and_log_train_metrics(
            train_model.losses, train_model.metrics,
            curr_iter, train_timer, rem_train_iters,
            cfg.SOLVER.NUM_ITERATIONS, mb_size
        )
        if (cfg.CHECKPOINT.CHECKPOINT_MODEL and
                (curr_iter + 1) % cfg.CHECKPOINT.CHECKPOINT_PERIOD == 0):
            params_file = os.path.join(
                checkpoint_dir, 'c2_model_iter{}.pkl'.format(curr_iter + 1))
            checkpoints_rel.save_model_params(
                model=train_model, params_file=params_file,
                model_iter=curr_iter, checkpoint_dir=checkpoint_dir
            )
            params_file = os.path.join(checkpoint_dir, 'latest.pkl')
            checkpoints_rel.save_model_params(
                model=train_model, params_file=params_file,
                model_iter=curr_iter, checkpoint_dir=checkpoint_dir
            )

        # do val
        if (curr_iter + 1) % cfg.TRAIN.EVALUATION_FREQUENCY == 0:
            train_metrics_calculator.finalize_metrics()
            if cfg.CHECKPOINT.CHECKPOINT_MODEL:
                params_file = os.path.join(checkpoint_dir, 'latest.pkl')
                checkpoints_rel.save_model_params(
                    model=train_model, params_file=params_file,
                    model_iter=curr_iter, checkpoint_dir=checkpoint_dir
                )

                checkpoints_rel.initialize_params_from_file(
                    model=val_model, weights_file=params_file,
                    num_devices=cfg.NUM_DEVICES,
                )
                val_evaluator.reset()
                print("=> Validating model")
                for _ in range(0, total_val_iters):
                    val_timer.tic()
                    workspace.RunNet(val_model.net.Proto().name)
                    val_timer.toc()
                    val_evaluator.eval_im_dets_triplet_topk()
                iter_accs = val_evaluator.calculate_and_plot_accuracy()
                for key in iter_accs.keys():
                    accumulated_accs[key].append(iter_accs[key])
                curr_avg_top1_acc = (iter_accs['rel_top1_acc'] + iter_accs['obj_top1_acc'] + iter_accs['sbj_top1_acc']) / 3.0
                best = json.load(open(os.path.join(checkpoint_dir, 'best.json')))
                if curr_avg_top1_acc > best['best_avg_top1_acc']:
                    print('Found new best validation accuracy at {}%'.format(curr_avg_top1_acc))
                    print('Saving best model..')
                    best['best_avg_top1_acc'] = curr_avg_top1_acc
                    best['iteration'] = curr_iter
                    best['accuracies'] = [iter_accs]
                    params_file = os.path.join(checkpoint_dir, 'best.pkl')
                    checkpoints_rel.save_model_params(
                        model=train_model, params_file=params_file,
                        model_iter=curr_iter, checkpoint_dir=checkpoint_dir
                    )
                    json.dump(best, open(os.path.join(checkpoint_dir, 'best.json'), 'w'))

    train_model.roi_data_loader.shutdown()
    if True:
        val_model.roi_data_loader.shutdown()

    logger.info('Training has successfully finished...exiting!')
