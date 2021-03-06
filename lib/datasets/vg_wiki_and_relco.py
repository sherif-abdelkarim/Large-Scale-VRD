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
from datasets.imdb_rel import imdb_rel
import utils.boxes as box_utils
import numpy as np
import scipy.sparse
import sparse
import json
import cPickle
from core.config_rel import cfg

import gensim
# from autocorrect import spell
from numpy import linalg as la
import PIL
import time
import logging
logger = logging.getLogger(__name__)


class vg_wiki_and_relco(imdb_rel):
    def __init__(self, image_set):
        if cfg.DATASET == 'gvqa':
            imdb_rel.__init__(self, 'gvqa_' + image_set)
        elif cfg.DATASET == 'vg_wiki_and_relco':
            imdb_rel.__init__(self, 'vg_wiki_and_relco_' + image_set)
        else:
            raise NotImplementedError
        self._image_set = image_set

        if cfg.DATASET == 'gvqa':
            self._data_path = os.path.join(cfg.DATA_DIR, 'GVQA')
        elif cfg.DATASET == 'vg_wiki_and_relco':
            self._data_path = os.path.join(cfg.DATA_DIR, 'Visual_Genome')
        else:
            raise NotImplementedError

        self._object_classes = ['__background__']
        with open(self._data_path + '/object_categories_spo_joined_and_merged.txt') as obj_classes:
            for line in obj_classes:
                self._object_classes.append(line[:-1])
        self._num_object_classes = len(self._object_classes)
        self._object_class_to_ind = \
            dict(zip(self._object_classes, range(self._num_object_classes)))
        logger.info(len(self._object_class_to_ind))
        # cfg.MODEL.NUM_CLASSES_SBJ_OBJ = self._num_object_classes -1

        self._predicate_classes = ['__background__']
        with open(self._data_path + '/predicate_categories_spo_joined_and_merged.txt') as prd_classes:
            for line in prd_classes:
                self._predicate_classes.append(line[:-1])
        self._num_predicate_classes = len(self._predicate_classes)
        self._predicate_class_to_ind = \
            dict(zip(self._predicate_classes, range(self._num_predicate_classes)))
        logger.info(len(self._predicate_class_to_ind))
        # cfg.MODEL.NUM_CLASSES_PRD = self._num_predicate_classes -1

        self._image_index = self._load_image_set_index()
        # Default to roidb handler
        self._roidb_handler = self.gt_roidb

        self.model = None
        self.relco_model = None
        self.relco_vec_mean = None

        assert os.path.exists(self._data_path), \
            'Path does not exist: {}'.format(self._data_path)

    def image_path_at(self, i):
        """
        Return the absolute path to image i in the image sequence.
        """
        return self.image_path_from_index(self._image_index[i])

    def image_path_from_index(self, index):
        """
        Construct an image path from the image's "index" identifier.
        """
        image_path = os.path.join(self._data_path, 'images', str(index) + '.jpg')
        assert os.path.exists(image_path), \
            'Path does not exist: {}'.format(image_path)
        return image_path

    def get_widths_and_heights(self):
        cache_file = os.path.join(
            self._data_path, 'vg_' + self._image_set + '_image_sizes{}.pkl'.format(cfg.RNG_SEED))
        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as fid:
                sizes = cPickle.load(fid)
            print('{} image sizes loaded from {}'.format(self.name, cache_file))
            return sizes[:, 0], sizes[:, 1]

        sizes_list = [None] * self.num_images
        for i in range(self.num_images):
            sizes_list[i] = PIL.Image.open(self.image_path_at(i)).size
            print('getting size for image ', i + 1)

        sizes = np.array(sizes_list)

        print('widths: ', sizes[:, 0])
        print('heights: ', sizes[:, 1])

        with open(cache_file, 'wb') as fid:
            cPickle.dump(sizes, fid, cPickle.HIGHEST_PROTOCOL)
        print('wrote image sizes to {}'.format(cache_file))

        return sizes[:, 0], sizes[:, 1]

    def _load_image_set_index(self):
        """
        Load the indexes listed in this dataset's image set file.
        """
        image_set_file = os.path.join(self._data_path, 'random_splits/seed{}'.format(cfg.RNG_SEED), self._image_set + '_clean.json')
        # image_set_file = os.path.join(self._data_path, self._image_set + '_clean.json')
        assert os.path.exists(image_set_file), \
            'Path does not exist: {}'.format(image_set_file)
        with open(image_set_file) as f:
            data = json.load(f)
        return data

    def gt_roidb(self):
        """
        Return the database of ground-truth regions of interest.

        This function loads/saves from/to a cache file to speed up future calls.
        """
        if cfg.MODEL.WEAK_LABELS:
            cache_file = os.path.join(self.cache_path, self.name + '_gt_roidb_w{}.pkl'.format(cfg.RNG_SEED))
        else:
            cache_file = os.path.join(self.cache_path, self.name + '_gt_roidb{}.pkl'.format(cfg.RNG_SEED))

        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as fid:
                roidb = cPickle.load(fid)
            logger.info('{} gt roidb loaded from {}'.format(self.name, cache_file))
            return roidb

        # Load gt data from scratch
        # Load Google's pre-trained Word2Vec model.
        self.model = gensim.models.KeyedVectors.load_word2vec_format(
            cfg.DATA_DIR + '/models/GoogleNews-vectors-negative300.bin', binary=True)
        print('Model loaded.')
        # change everything into lowercase
        for key in self.model.vocab.keys():
            new_key = key.lower()
            # new_key = new_key.replace('-', '_')
            self.model.vocab[new_key] = self.model.vocab.pop(key)
        # self.model.vocab['t-shirt'] = self.model.vocab['t_shirt']
        temp_dict = {x.replace('_', '-'): y for x, y in self.model.vocab.items()}
        self.model.vocab.update(temp_dict)
        print('Wiki words converted to lowercase.')

        # Load gt data from scratch
        # Load Yannis' rel_cooccur_300d model.
        self.relco_model = gensim.models.Word2Vec.load(
            cfg.DATA_DIR + '/label_embeddings/vg_300d_skipgram_rel')
        print('Model loaded.')
        self.relco_vec_mean = self.relco_model.wv.syn0.mean(axis=0)
        # change everything into lowercase
        # for key in self.model.wv.vocab.keys():
        for key in self.relco_model.wv.vocab.keys():
            new_key = key.lower()
            self.relco_model.wv.vocab[new_key] = self.relco_model.wv.vocab.pop(key)
        temp_dict = {x.replace('_', '-'): y for x, y in self.relco_model.wv.vocab.items()}
        self.relco_model.wv.vocab.update(temp_dict)

        print('Relco words converted to lowercase.')

        rel_data_path = os.path.join(
            self._data_path, 'relationships_clean_spo_joined_and_merged.json')
        with open(rel_data_path) as f:
            all_rels = json.load(f)

        if cfg.MODEL.WEAK_LABELS:
            rel_w_data_path = os.path.join(
                self._data_path, 'relationships_weak_labels_top4.json')
            with open(rel_w_data_path) as f:
                rels_w = json.load(f)

        all_rels_map = {}
        for cnt, rel in enumerate(all_rels):
            all_rels_map[rel['image_id']] = cnt

        if cfg.MODEL.WEAK_LABELS:
            gt_roidb = \
                [self._load_vg_annotation_with_weak_labels(all_rels[all_rels_map[index]], rels_w[all_rels_map[index]],
                                                           index, cnt, len(self.image_index))
                 for cnt, index in enumerate(self.image_index)]

        else:
            gt_roidb = \
                [self._load_vg_annotation(all_rels[all_rels_map[index]],
                                          index, cnt, len(self.image_index))
                 for cnt, index in enumerate(self.image_index)]

        with open(cache_file, 'wb') as fid:
            cPickle.dump(gt_roidb, fid, cPickle.HIGHEST_PROTOCOL)
        print('wrote gt roidb to {}'.format(cache_file))

        return gt_roidb

    def _load_vg_annotation(self, img_rels, index, cnt, length):
        """
        Load image and bounding boxes info.
        """

        print("Loading image %d/%d..." % (cnt + 1, length))

        assert index == img_rels['image_id']  # sanity check

        num_rels = len(img_rels['relationships'])

        sbj_boxes = np.zeros((num_rels, 4), dtype=np.uint16)
        obj_boxes = np.zeros((num_rels, 4), dtype=np.uint16)
        rel_boxes = np.zeros((num_rels, 4), dtype=np.uint16)
        sbj_names = np.zeros((num_rels), dtype='U100')
        obj_names = np.zeros((num_rels), dtype='U100')
        prd_names = np.zeros((num_rels), dtype='U100')
        gt_sbj_classes = np.zeros((num_rels), dtype=np.int32)
        gt_obj_classes = np.zeros((num_rels), dtype=np.int32)
        gt_rel_classes = np.zeros((num_rels), dtype=np.int32)
        sbj_overlaps = \
            np.zeros((num_rels, self._num_object_classes), dtype=np.float32)
        obj_overlaps = \
            np.zeros((num_rels, self._num_object_classes), dtype=np.float32)
        rel_overlaps = \
            np.zeros((num_rels, self._num_predicate_classes), dtype=np.float32)
        # "Seg" area for pascal is just the box area
        sbj_seg_areas = np.zeros((num_rels), dtype=np.float32)
        obj_seg_areas = np.zeros((num_rels), dtype=np.float32)
        rel_seg_areas = np.zeros((num_rels), dtype=np.float32)

        # variables for word vectors
        half_dim = int(cfg.INPUT_LANG_EMBEDDING_DIM / 2)
        sbj_vecs = np.zeros(
            (num_rels, cfg.INPUT_LANG_EMBEDDING_DIM), dtype=np.float32)
        obj_vecs = np.zeros(
            (num_rels, cfg.INPUT_LANG_EMBEDDING_DIM), dtype=np.float32)
        prd_vecs = np.zeros(
            (num_rels, cfg.INPUT_LANG_EMBEDDING_DIM), dtype=np.float32)
        # Load object bounding boxes into a data frame.
        for ix, rel in enumerate(img_rels['relationships']):
            sbj = rel['subject']
            obj = rel['object']
            prd = rel['predicate']
            sbj_box = [sbj['x'], sbj['y'], sbj['x'] + sbj['w'], sbj['y'] + sbj['h']]
            obj_box = [obj['x'], obj['y'], obj['x'] + obj['w'], obj['y'] + obj['h']]
            rel_box = box_utils.box_union(sbj_box, obj_box)
            sbj_boxes[ix, :] = sbj_box
            obj_boxes[ix, :] = obj_box
            rel_boxes[ix, :] = rel_box
            sbj_names[ix] = sbj['name']
            obj_names[ix] = obj['name']
            prd_names[ix] = prd
            sbj_cls = self._object_class_to_ind[str(sbj_names[ix])]
            obj_cls = self._object_class_to_ind[str(obj_names[ix])]
            prd_cls = self._predicate_class_to_ind[str(prd_names[ix])]
            gt_sbj_classes[ix] = sbj_cls
            gt_obj_classes[ix] = obj_cls
            gt_rel_classes[ix] = prd_cls
            sbj_overlaps[ix, sbj_cls] = 1.0
            obj_overlaps[ix, obj_cls] = 1.0
            rel_overlaps[ix, prd_cls] = 1.0
            sbj_seg_areas[ix] = (sbj_box[2] - sbj_box[0] + 1) * \
                                (sbj_box[3] - sbj_box[1] + 1)
            obj_seg_areas[ix] = (obj_box[2] - obj_box[0] + 1) * \
                                (obj_box[3] - obj_box[1] + 1)
            rel_seg_areas[ix] = (rel_box[2] - rel_box[0] + 1) * \
                                (rel_box[3] - rel_box[1] + 1)

            # add word vectors for sbjs, objs and rels
            # sbj word2vec
            sbj_vecs_wiki = np.zeros(half_dim, dtype=np.float32)
            sbj_words = sbj_names[ix].split()
            for word in sbj_words:
                if word in self.model.vocab:
                    raw_word = self.model[word]
                    sbj_vecs_wiki += (raw_word / la.norm(raw_word))
                else:
                    print('Singular word found: ', word)
                    raise NameError('Terminated.')
            sbj_vecs_wiki /= len(sbj_words)
            sbj_vecs_wiki /= la.norm(sbj_vecs_wiki)

            sbj_vecs_relco = np.zeros(half_dim, dtype=np.float32)
            sbj_words = sbj_names[ix].split()
            for word in sbj_words:
                if word in self.relco_model.wv.vocab:
                    raw_word = self.relco_model[word]
                    sbj_vecs_relco += (raw_word / la.norm(raw_word))
                else:
                    sbj_vecs_relco += \
                        (self.relco_vec_mean / la.norm(self.relco_vec_mean))
            sbj_vecs_relco /= len(sbj_words)
            sbj_vecs_relco /= la.norm(sbj_vecs_relco)

            sbj_vecs[ix][:half_dim] = sbj_vecs_wiki
            sbj_vecs[ix][half_dim:] = sbj_vecs_relco

            # obj word2vec
            obj_vecs_wiki = np.zeros(half_dim, dtype=np.float32)
            obj_words = obj_names[ix].split()
            for word in obj_words:
                if word in self.model.vocab:
                    raw_word = self.model[word]
                    obj_vecs_wiki += (raw_word / la.norm(raw_word))
                else:
                    print('Singular word found: ', word)
                    raise NameError('Terminated.')
            obj_vecs_wiki /= len(obj_words)
            obj_vecs_wiki /= la.norm(obj_vecs_wiki)

            obj_vecs_relco = np.zeros(half_dim, dtype=np.float32)
            obj_words = obj_names[ix].split()
            for word in obj_words:
                if word in self.relco_model.wv.vocab:
                    raw_word = self.relco_model[word]
                    obj_vecs_relco += (raw_word / la.norm(raw_word))
                else:
                    obj_vecs_relco += \
                        (self.relco_vec_mean / la.norm(self.relco_vec_mean))
            obj_vecs_relco /= len(obj_words)
            obj_vecs_relco /= la.norm(obj_vecs_relco)

            obj_vecs[ix][:half_dim] = obj_vecs_wiki
            obj_vecs[ix][half_dim:] = obj_vecs_relco

            # prd word2vec
            prd_vecs_wiki = np.zeros(half_dim, dtype=np.float32)
            prd_words = prd_names[ix].split()
            for word in prd_words:
                if word in self.model.vocab:
                    raw_word = self.model[word]
                    prd_vecs_wiki += (raw_word / la.norm(raw_word))
                else:
                    print('Singular word found: ', word)
                    raise NameError('Terminated.')
            prd_vecs_wiki /= len(prd_words)
            prd_vecs_wiki /= la.norm(prd_vecs_wiki)

            prd_vecs_relco = np.zeros(half_dim, dtype=np.float32)
            prd_words = prd_names[ix].split()
            for word in prd_words:
                if word in self.relco_model.wv.vocab:
                    raw_word = self.relco_model[word]
                    prd_vecs_relco += (raw_word / la.norm(raw_word))
                else:
                    prd_vecs_relco += \
                        (self.relco_vec_mean / la.norm(self.relco_vec_mean))
            prd_vecs_relco /= len(prd_words)
            prd_vecs_relco /= la.norm(prd_vecs_relco)

            prd_vecs[ix][:half_dim] = prd_vecs_wiki
            prd_vecs[ix][half_dim:] = prd_vecs_relco

        sbj_overlaps = scipy.sparse.csr_matrix(sbj_overlaps)
        obj_overlaps = scipy.sparse.csr_matrix(obj_overlaps)
        rel_overlaps = scipy.sparse.csr_matrix(rel_overlaps)
        return {'sbj_boxes': sbj_boxes,
                'obj_boxes': obj_boxes,
                'rel_boxes': rel_boxes,
                'sbj_names': sbj_names,
                'obj_names': obj_names,
                'prd_names': prd_names,
                'gt_sbj_classes': gt_sbj_classes,
                'gt_obj_classes': gt_obj_classes,
                'gt_rel_classes': gt_rel_classes,
                'gt_sbj_overlaps': sbj_overlaps,
                'gt_obj_overlaps': obj_overlaps,
                'gt_rel_overlaps': rel_overlaps,
                'sbj_seg_areas': sbj_seg_areas,
                'obj_seg_areas': obj_seg_areas,
                'rel_seg_areas': rel_seg_areas,
                'sbj_vecs': sbj_vecs,
                'obj_vecs': obj_vecs,
                'prd_vecs': prd_vecs,
                'flipped': False}

    def _load_vg_annotation_with_weak_labels(self, img_rels, img_rels_w, index, cnt, length):
        """
        Load image and bounding boxes info.
        """

        print("Loading image %d/%d..." % (cnt + 1, length))
        assert index == img_rels['image_id']  # sanity check

        num_rels = len(img_rels['relationships'])

        sbj_boxes = np.zeros((num_rels, 4), dtype=np.uint16)
        obj_boxes = np.zeros((num_rels, 4), dtype=np.uint16)
        rel_boxes = np.zeros((num_rels, 4), dtype=np.uint16)
        sbj_names = np.zeros((num_rels), dtype='U100')
        obj_names = np.zeros((num_rels), dtype='U100')
        prd_names = np.zeros((num_rels), dtype='U100')
        sbj_names_w = np.zeros((num_rels, cfg.MODEL.NUM_WEAK_LABELS), dtype='U100')
        obj_names_w = np.zeros((num_rels, cfg.MODEL.NUM_WEAK_LABELS), dtype='U100')
        prd_names_w = np.zeros((num_rels, cfg.MODEL.NUM_WEAK_LABELS), dtype='U100')
        gt_sbj_classes = np.zeros((num_rels), dtype=np.int32)
        gt_obj_classes = np.zeros((num_rels), dtype=np.int32)
        gt_rel_classes = np.zeros((num_rels), dtype=np.int32)
        gt_sbj_classes_w = np.zeros((num_rels, cfg.MODEL.NUM_WEAK_LABELS), dtype=np.int32)
        gt_obj_classes_w = np.zeros((num_rels, cfg.MODEL.NUM_WEAK_LABELS), dtype=np.int32)
        gt_rel_classes_w = np.zeros((num_rels, cfg.MODEL.NUM_WEAK_LABELS), dtype=np.int32)
        sbj_overlaps = \
            np.zeros((num_rels, self._num_object_classes), dtype=np.float32)
        obj_overlaps = \
            np.zeros((num_rels, self._num_object_classes), dtype=np.float32)
        rel_overlaps = \
            np.zeros((num_rels, self._num_predicate_classes), dtype=np.float32)

        sbj_overlaps_w = \
            np.zeros((num_rels, cfg.MODEL.NUM_WEAK_LABELS, self._num_object_classes), dtype=np.float32)
        obj_overlaps_w = \
            np.zeros((num_rels, cfg.MODEL.NUM_WEAK_LABELS, self._num_object_classes), dtype=np.float32)
        rel_overlaps_w = \
            np.zeros((num_rels, cfg.MODEL.NUM_WEAK_LABELS, self._num_predicate_classes), dtype=np.float32)

        # "Seg" area for pascal is just the box area
        sbj_seg_areas = np.zeros((num_rels), dtype=np.float32)
        obj_seg_areas = np.zeros((num_rels), dtype=np.float32)
        rel_seg_areas = np.zeros((num_rels), dtype=np.float32)

        # variables for word vectors
        half_dim = int(cfg.INPUT_LANG_EMBEDDING_DIM / 2)
        sbj_vecs = np.zeros(
            (num_rels, cfg.INPUT_LANG_EMBEDDING_DIM), dtype=np.float32)
        obj_vecs = np.zeros(
            (num_rels, cfg.INPUT_LANG_EMBEDDING_DIM), dtype=np.float32)
        prd_vecs = np.zeros(
            (num_rels, cfg.INPUT_LANG_EMBEDDING_DIM), dtype=np.float32)

        # {'predicate_weak_labels': ['in', 'of', 'on side of', 'next to'],
        #  'subject': {'name_weak_labels': ['sidewalk', 'ground', 'shadow', 'pavement']},
        #  'object': {'name_weak_labels': ['street', 'road', 'ground', 'pavement']}}
        # Load object bounding boxes into a data frame.
        for ix, rel in enumerate(img_rels['relationships']):
            sbj = rel['subject']
            obj = rel['object']
            prd = rel['predicate']

            sbj_w = img_rels_w['relationships'][ix]['subject']
            obj_w = img_rels_w['relationships'][ix]['object']
            prd_w = img_rels_w['relationships'][ix]['predicate_weak_labels']

            sbj_box = [sbj['x'], sbj['y'], sbj['x'] + sbj['w'], sbj['y'] + sbj['h']]
            obj_box = [obj['x'], obj['y'], obj['x'] + obj['w'], obj['y'] + obj['h']]
            rel_box = box_utils.box_union(sbj_box, obj_box)
            sbj_boxes[ix, :] = sbj_box
            obj_boxes[ix, :] = obj_box
            rel_boxes[ix, :] = rel_box

            sbj_names[ix] = sbj['name']
            obj_names[ix] = obj['name']
            prd_names[ix] = prd

            sbj_names_w[ix] = sbj_w['name_weak_labels']
            obj_names_w[ix] = obj_w['name_weak_labels']
            prd_names_w[ix] = prd_w

            sbj_cls = self._object_class_to_ind[str(sbj_names[ix])]
            obj_cls = self._object_class_to_ind[str(obj_names[ix])]
            prd_cls = self._predicate_class_to_ind[str(prd_names[ix])]

            sbj_cls_w = np.array([self._object_class_to_ind[str(sbj_names_w[ix][i])] for i in range(len(sbj_names_w[ix]))], dtype=np.int32)
            obj_cls_w = np.array([self._object_class_to_ind[str(obj_names_w[ix][i])] for i in range(len(sbj_names_w[ix]))], dtype=np.int32)
            prd_cls_w = np.array([self._predicate_class_to_ind[str(prd_names_w[ix][i])] for i in range(len(sbj_names_w[ix]))], dtype=np.int32)

            gt_sbj_classes[ix] = sbj_cls
            gt_obj_classes[ix] = obj_cls
            gt_rel_classes[ix] = prd_cls

            gt_sbj_classes_w[ix, :] = sbj_cls_w
            gt_obj_classes_w[ix, :] = obj_cls_w
            gt_rel_classes_w[ix, :] = prd_cls_w

            sbj_overlaps[ix, sbj_cls] = 1.0
            obj_overlaps[ix, obj_cls] = 1.0
            rel_overlaps[ix, prd_cls] = 1.0

            for i, s in enumerate(sbj_cls_w):
                sbj_overlaps_w[ix, i, s] = 1.0
            for i, o in enumerate(obj_cls_w):
                obj_overlaps_w[ix, i, o] = 1.0
            for i, p in enumerate(prd_cls_w):
                rel_overlaps_w[ix, i, p] = 1.0

            sbj_seg_areas[ix] = (sbj_box[2] - sbj_box[0] + 1) * \
                                (sbj_box[3] - sbj_box[1] + 1)
            obj_seg_areas[ix] = (obj_box[2] - obj_box[0] + 1) * \
                                (obj_box[3] - obj_box[1] + 1)
            rel_seg_areas[ix] = (rel_box[2] - rel_box[0] + 1) * \
                                (rel_box[3] - rel_box[1] + 1)

            # add word vectors for sbjs, objs and rels
            # sbj word2vec
            sbj_vecs_wiki = np.zeros(half_dim, dtype=np.float32)
            sbj_words = sbj_names[ix].split()
            for word in sbj_words:
                if word in self.model.vocab:
                    raw_word = self.model[word]
                    sbj_vecs_wiki += (raw_word / la.norm(raw_word))
                else:
                    print('Singular word found: ', word)
                    raise NameError('Terminated.')
            sbj_vecs_wiki /= len(sbj_words)
            sbj_vecs_wiki /= la.norm(sbj_vecs_wiki)

            sbj_vecs_relco = np.zeros(half_dim, dtype=np.float32)
            sbj_words = sbj_names[ix].split()
            for word in sbj_words:
                if word in self.relco_model.wv.vocab:
                    raw_word = self.relco_model[word]
                    sbj_vecs_relco += (raw_word / la.norm(raw_word))
                else:
                    sbj_vecs_relco += \
                        (self.relco_vec_mean / la.norm(self.relco_vec_mean))
            sbj_vecs_relco /= len(sbj_words)
            sbj_vecs_relco /= la.norm(sbj_vecs_relco)

            sbj_vecs[ix][:half_dim] = sbj_vecs_wiki
            sbj_vecs[ix][half_dim:] = sbj_vecs_relco

            # obj word2vec
            obj_vecs_wiki = np.zeros(half_dim, dtype=np.float32)
            obj_words = obj_names[ix].split()
            for word in obj_words:
                if word in self.model.vocab:
                    raw_word = self.model[word]
                    obj_vecs_wiki += (raw_word / la.norm(raw_word))
                else:
                    print('Singular word found: ', word)
                    raise NameError('Terminated.')
            obj_vecs_wiki /= len(obj_words)
            obj_vecs_wiki /= la.norm(obj_vecs_wiki)

            obj_vecs_relco = np.zeros(half_dim, dtype=np.float32)
            obj_words = obj_names[ix].split()
            for word in obj_words:
                if word in self.relco_model.wv.vocab:
                    raw_word = self.relco_model[word]
                    obj_vecs_relco += (raw_word / la.norm(raw_word))
                else:
                    obj_vecs_relco += \
                        (self.relco_vec_mean / la.norm(self.relco_vec_mean))
            obj_vecs_relco /= len(obj_words)
            obj_vecs_relco /= la.norm(obj_vecs_relco)

            obj_vecs[ix][:half_dim] = obj_vecs_wiki
            obj_vecs[ix][half_dim:] = obj_vecs_relco

            # prd word2vec
            prd_vecs_wiki = np.zeros(half_dim, dtype=np.float32)
            prd_words = prd_names[ix].split()
            for word in prd_words:
                if word in self.model.vocab:
                    raw_word = self.model[word]
                    prd_vecs_wiki += (raw_word / la.norm(raw_word))
                else:
                    print('Singular word found: ', word)
                    raise NameError('Terminated.')
            prd_vecs_wiki /= len(prd_words)
            prd_vecs_wiki /= la.norm(prd_vecs_wiki)

            prd_vecs_relco = np.zeros(half_dim, dtype=np.float32)
            prd_words = prd_names[ix].split()
            for word in prd_words:
                if word in self.relco_model.wv.vocab:
                    raw_word = self.relco_model[word]
                    prd_vecs_relco += (raw_word / la.norm(raw_word))
                else:
                    prd_vecs_relco += \
                        (self.relco_vec_mean / la.norm(self.relco_vec_mean))
            prd_vecs_relco /= len(prd_words)
            prd_vecs_relco /= la.norm(prd_vecs_relco)

            prd_vecs[ix][:half_dim] = prd_vecs_wiki
            prd_vecs[ix][half_dim:] = prd_vecs_relco


        sbj_overlaps = scipy.sparse.csr_matrix(sbj_overlaps)
        obj_overlaps = scipy.sparse.csr_matrix(obj_overlaps)
        rel_overlaps = scipy.sparse.csr_matrix(rel_overlaps)

        sbj_overlaps_w = sparse.COO(sbj_overlaps_w)
        obj_overlaps_w = sparse.COO(obj_overlaps_w)
        rel_overlaps_w = sparse.COO(rel_overlaps_w)

        return {'sbj_boxes': sbj_boxes,
                'obj_boxes': obj_boxes,
                'rel_boxes': rel_boxes,
                'sbj_names': sbj_names,
                'obj_names': obj_names,
                'prd_names': prd_names,
                'sbj_names_w': sbj_names_w,
                'obj_names_w': obj_names_w,
                'prd_names_w': prd_names_w,
                'gt_sbj_classes': gt_sbj_classes,
                'gt_obj_classes': gt_obj_classes,
                'gt_rel_classes': gt_rel_classes,
                'gt_sbj_classes_w': gt_sbj_classes_w,
                'gt_obj_classes_w': gt_obj_classes_w,
                'gt_rel_classes_w': gt_rel_classes_w,
                'gt_sbj_overlaps': sbj_overlaps,
                'gt_obj_overlaps': obj_overlaps,
                'gt_rel_overlaps': rel_overlaps,
                'gt_sbj_overlaps_w': sbj_overlaps_w,
                'gt_obj_overlaps_w': obj_overlaps_w,
                'gt_rel_overlaps_w': rel_overlaps_w,
                'sbj_seg_areas': sbj_seg_areas,
                'obj_seg_areas': obj_seg_areas,
                'rel_seg_areas': rel_seg_areas,
                'sbj_vecs': sbj_vecs,
                'obj_vecs': obj_vecs,
                'prd_vecs': prd_vecs,
                'flipped': False}
