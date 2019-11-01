import pickle
import numpy as np
from tqdm import tqdm
import cPickle

centroids_obj = np.zeros((1703, 1024), dtype=np.float64)
centroids_rel = np.zeros((310, 1024), dtype=np.float64)

print('Loading embeddings')
sbj_embs = pickle.load(open('/mnt/scratch/kwc/vision/Mohamed/large_scale_VRD.caffe2/sherif_github/Large-Scale-VRD/checkpoints/gvqa/VGG16_reldn_fast_rcnn_conv4_spo_for_p/embd_fusion_w_relu_yall/8gpus_vgg16_softmaxed_triplet_no_last_l2norm_trainval_w_cluster_2_lan_layers/train/all_sbj_vis_embds.pkl', 'r'))
obj_embs = pickle.load(open('/mnt/scratch/kwc/vision/Mohamed/large_scale_VRD.caffe2/sherif_github/Large-Scale-VRD/checkpoints/gvqa/VGG16_reldn_fast_rcnn_conv4_spo_for_p/embd_fusion_w_relu_yall/8gpus_vgg16_softmaxed_triplet_no_last_l2norm_trainval_w_cluster_2_lan_layers/train/all_obj_vis_embds.pkl', 'r'))
prd_embs = pickle.load(open('/mnt/scratch/kwc/vision/Mohamed/large_scale_VRD.caffe2/sherif_github/Large-Scale-VRD/checkpoints/gvqa/VGG16_reldn_fast_rcnn_conv4_spo_for_p/embd_fusion_w_relu_yall/8gpus_vgg16_softmaxed_triplet_no_last_l2norm_trainval_w_cluster_2_lan_layers/train/all_prd_vis_embds.pkl', 'r'))
print('Loading detections')
detections =  pickle.load(open('/mnt/scratch/kwc/vision/Mohamed/large_scale_VRD.caffe2/sherif_github/Large-Scale-VRD/checkpoints/gvqa/VGG16_reldn_fast_rcnn_conv4_spo_for_p/embd_fusion_w_relu_yall/8gpus_vgg16_softmaxed_triplet_no_last_l2norm_trainval_w_cluster_2_lan_layers/train/reldn_detections.pkl', 'r'))

print('Loading frequencies')
freq_obj = np.load('/mnt/scratch/kwc/vision/Mohamed/GVQA/freq_obj.npy')
freq_pred = np.load('/mnt/scratch/kwc/vision/Mohamed/GVQA/freq_pred.npy')
for i in tqdm(range(len(detections['image_idx']))):
    #print(detections['gt_labels_sbj'][i].shape)
    #print(sbj_embs[i].shape)
    #print('___________________________________')
    for j in range(len(detections['gt_labels_sbj'][i])):
        centroids_obj[detections['gt_labels_sbj'][i][j]] += sbj_embs[i][j]
        centroids_obj[detections['gt_labels_obj'][i][j]] += obj_embs[i][j]
        centroids_rel[detections['gt_labels_rel'][i][j]] += prd_embs[i][j]
freq_obj[np.where(freq_obj==0)] = 1
freq_pred[np.where(freq_pred==0)] = 1
freq_obj = np.expand_dims(freq_obj, axis=-1)
freq_pred = np.expand_dims(freq_pred, axis=-1)
print('freq_obj', freq_obj.shape)
print('freq_rel', freq_pred.shape)
print('centroids_obj', centroids_obj.shape)
print('centroids_rel', centroids_rel.shape)
centroids_obj = centroids_obj / freq_obj
centroids_rel = centroids_rel / freq_pred
print('Saving pickles')
pickle.dump(centroids_obj, open('./centroids_obj.pkl', 'w'), cPickle.HIGHEST_PROTOCOL)
pickle.dump(centroids_rel, open('./centroids_rel.pkl', 'w'), cPickle.HIGHEST_PROTOCOL)
