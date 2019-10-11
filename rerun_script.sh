#!/bin/bash

for (( ; ; ))
do
	# Baseline
	#docker pull dockerhub.svail.baidu.com/lsvrd_caffe2 && nvidia-docker run --rm -v /mnt/scratch/:/mnt/scratch/:rw -it dockerhub.svail.baidu.com/lsvrd_caffe2:latest /bin/bash -c "cd /mnt/scratch/kwc/vision/Mohamed/large_scale_VRD.caffe2/sherif_github/Large-Scale-VRD/ ; ls -ahl; cd lib/; make; cd ..; python tools/train_net_rel.py --cfg configs/vg/GVQA_VGG16_softmaxed_triplet_2_lan_layers_8gpu.yaml 2>&1 | tee logs/`date '+%Y_%m_%d__%H_%M_%S'`.txt"
	# Hubness
	docker pull dockerhub.svail.baidu.com/lsvrd_caffe2 && nvidia-docker run --rm -v /mnt/scratch/:/mnt/scratch/:rw -it dockerhub.svail.baidu.com/lsvrd_caffe2:latest /bin/bash -c "cd /mnt/scratch/kwc/vision/Mohamed/large_scale_VRD.caffe2/sherif_github/Large-Scale-VRD/ ; ls -ahl; cd lib/; make; cd ..; python tools/train_net_rel.py --cfg configs/vg/GVQA_VGG16_softmaxed_triplet_2_lan_layers_hubness_8gpu.yaml 2>&1 | tee logs/`date '+%Y_%m_%d__%H_%M_%S'`.txt"
	# Hubness 10k
	#docker pull dockerhub.svail.baidu.com/lsvrd_caffe2 && nvidia-docker run --rm -v /mnt/scratch/:/mnt/scratch/:rw -it dockerhub.svail.baidu.com/lsvrd_caffe2:latest /bin/bash -c "cd /mnt/scratch/kwc/vision/Mohamed/large_scale_VRD.caffe2/sherif_github/Large-Scale-VRD/ ; ls -ahl; cd lib/; make; cd ..; python tools/train_net_rel.py --cfg configs/vg/GVQA_VGG16_softmaxed_triplet_2_lan_layers_hubness10000_8gpu.yaml 2>&1 | tee logs/`date '+%Y_%m_%d__%H_%M_%S'`.txt"	
	# Hubness 50k
	#docker pull dockerhub.svail.baidu.com/lsvrd_caffe2 && nvidia-docker run --rm -v /mnt/scratch/:/mnt/scratch/:rw -it dockerhub.svail.baidu.com/lsvrd_caffe2:latest /bin/bash -c "cd /mnt/scratch/kwc/vision/Mohamed/large_scale_VRD.caffe2/sherif_github/Large-Scale-VRD/ ; ls -ahl; cd lib/; make; cd ..; python tools/train_net_rel.py --cfg configs/vg/GVQA_VGG16_softmaxed_triplet_2_lan_layers_hubness50000_8gpu.yaml 2>&1 | tee logs/`date '+%Y_%m_%d__%H_%M_%S'`.txt"	
	# Focal Loss
	#docker pull dockerhub.svail.baidu.com/lsvrd_caffe2 && nvidia-docker run --rm -v /mnt/scratch/:/mnt/scratch/:rw -it dockerhub.svail.baidu.com/lsvrd_caffe2:latest /bin/bash -c "cd /mnt/scratch/kwc/vision/Mohamed/large_scale_VRD.caffe2/sherif_github/Large-Scale-VRD/ ; ls -ahl; cd lib/; make; cd ..; python tools/train_net_rel.py --cfg configs/vg/GVQA_VGG16_softmaxed_triplet_2_lan_layers_focal_loss_8gpu.yaml 2>&1 | tee logs/`date '+%Y_%m_%d__%H_%M_%S'`.txt" 
	# Focal Loss Gamma 5
	#docker pull dockerhub.svail.baidu.com/lsvrd_caffe2 && nvidia-docker run --rm -v /mnt/scratch/:/mnt/scratch/:rw -it dockerhub.svail.baidu.com/lsvrd_caffe2:latest /bin/bash -c "cd /mnt/scratch/kwc/vision/Mohamed/large_scale_VRD.caffe2/sherif_github/Large-Scale-VRD/ ; ls -ahl; cd lib/; make; cd ..; python tools/train_net_rel.py --cfg configs/vg/GVQA_VGG16_softmaxed_triplet_2_lan_layers_focal_loss_g5_8gpu.yaml 2>&1 | tee logs/`date '+%Y_%m_%d__%H_%M_%S'`.txt"
	# Focal Loss Gamma 0.25
	#docker pull dockerhub.svail.baidu.com/lsvrd_caffe2 && nvidia-docker run --rm -v /mnt/scratch/:/mnt/scratch/:rw -it dockerhub.svail.baidu.com/lsvrd_caffe2:latest /bin/bash -c "cd /mnt/scratch/kwc/vision/Mohamed/large_scale_VRD.caffe2/sherif_github/Large-Scale-VRD/ ; ls -ahl; cd lib/; make; cd ..; python tools/train_net_rel.py --cfg configs/vg/GVQA_VGG16_softmaxed_triplet_2_lan_layers_focal_loss_g025_8gpu.yaml 2>&1 | tee logs/`date '+%Y_%m_%d__%H_%M_%S'`.txt"
	# Focal Loss Gamma 10
	#docker pull dockerhub.svail.baidu.com/lsvrd_caffe2 && nvidia-docker run --rm -v /mnt/scratch/:/mnt/scratch/:rw -it dockerhub.svail.baidu.com/lsvrd_caffe2:latest /bin/bash -c "cd /mnt/scratch/kwc/vision/Mohamed/large_scale_VRD.caffe2/sherif_github/Large-Scale-VRD/ ; ls -ahl; cd lib/; make; cd ..; python tools/train_net_rel.py --cfg configs/vg/GVQA_VGG16_softmaxed_triplet_2_lan_layers_focal_loss_g10_8gpu.yaml 2>&1 | tee logs/`date '+%Y_%m_%d__%H_%M_%S'`.txt"
	docker kill $(docker ps -q)
done
