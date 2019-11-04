import pickle
import numpy as np
from tqdm import tqdm
import cPickle

centroids_obj = np.zeros((1703, 1024), dtype=np.float64)
centroids_rel = np.zeros((310, 1024), dtype=np.float64)
#checkpoint_path = '/ibex/scratch/x_abdelks/Large-Scale-VRD/checkpoints/gvqa/VGG16_reldn_fast_rcnn_conv4_spo_for_p/embd_fusion_w_relu_yall/8gpus_vgg16_softmaxed_triplet_no_last_l2norm_trainval_w_cluster_2_lan_layers/train/'
checkpoint_path = '/ibex/scratch/x_abdelks/Large-Scale-VRD/checkpoints/gvqa/VGG16_reldn_fast_rcnn_conv4_spo_for_p/embd_fusion_w_relu_yall/8gpus_vgg16_softmax_no_last_l2norm_trainval_w_cluster_2_lan_layers/train/'
print('Loading embeddings')
sbj_embs = pickle.load(open(checkpoint_path + 'all_sbj_vis_embds.pkl', 'r'))
obj_embs = pickle.load(open(checkpoint_path + 'all_obj_vis_embds.pkl', 'r'))
prd_embs = pickle.load(open(checkpoint_path + 'all_prd_vis_embds.pkl', 'r'))
print('Loading detections')
detections =  pickle.load(open(checkpoint_path + 'reldn_detections.pkl', 'r'))

#print('Loading frequencies')
#freq_obj = np.load('/mnt/scratch/kwc/vision/Mohamed/GVQA/freq_obj.npy')
#freq_prd = np.load('/mnt/scratch/kwc/vision/Mohamed/GVQA/freq_prd.npy')

freq_obj = np.zeros((1703, 1), dtype=np.int64)
freq_prd = np.zeros((310, 1), dtype=np.int64)

for i in tqdm(range(len(detections['image_idx']))):
    #print(detections['gt_labels_sbj'][i].shape)
    #print(sbj_embs[i].shape)
    #print('___________________________________')
    for j in range(len(detections['gt_labels_sbj'][i])):
        centroids_obj[detections['gt_labels_sbj'][i][j]] += sbj_embs[i][j]
        centroids_obj[detections['gt_labels_obj'][i][j]] += obj_embs[i][j]
        centroids_rel[detections['gt_labels_rel'][i][j]] += prd_embs[i][j]
        freq_obj[detections['gt_labels_sbj'][i][j]] += 1
        freq_obj[detections['gt_labels_obj'][i][j]] += 1
        freq_prd[detections['gt_labels_rel'][i][j]] += 1

freq_obj[np.where(freq_obj==0)] = 1
freq_prd[np.where(freq_prd==0)] = 1
#freq_obj = np.expand_dims(freq_obj, axis=-1)
#freq_prd = np.expand_dims(freq_prd, axis=-1)
print('freq_obj', freq_obj.shape, 'zeros', freq_obj.size - np.count_nonzero(freq_obj))
print('freq_rel', freq_prd.shape, 'zeros', freq_prd.size - np.count_nonzero(freq_prd))
print('centroids_obj', centroids_obj.shape, 'zeros', centroids_obj.size - np.count_nonzero(centroids_obj))
print('centroids_rel', centroids_rel.shape, 'zeros', centroids_rel.size - np.count_nonzero(centroids_rel))
print('centroids_obj', centroids_obj[freq_obj[freq_obj==0], :])
centroids_obj = centroids_obj / freq_obj
centroids_rel = centroids_rel / freq_prd
print('Saving pickles')
pickle.dump(centroids_obj, open(checkpoint_path + 'centroids_obj.pkl', 'w'), cPickle.HIGHEST_PROTOCOL)
pickle.dump(centroids_rel, open(checkpoint_path + 'centroids_rel.pkl', 'w'), cPickle.HIGHEST_PROTOCOL)
