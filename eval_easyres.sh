# 20221102

# MOCO nolabel
# extract weights
python ./tools/model_converters/extract_backbone_weights.py work_dirs/selfsup/InUse/moco_resnet18_epoch1000_temp0_07_DIV2K/epoch_1000.pth work_dirs/selfsup/InUse/moco_resnet18_epoch1000_temp0_07_DIV2K/weights_1000.pth

# transfer learning
bash ./tools/benchmarks/classification/dist_train_linear.sh configs/benchmarks/classification/imagenet/resnet18_DIV2K.py work_dirs/selfsup/InUse/moco_resnet18_epoch1000_temp0_07_DIV2K/weights_1000.pth work_dirs/selfsup/InUse/transfer_5cls_moco_resnet18_epoch1000_temp0_07_DIV2K/

# for more cls
# not run yet
bash ./tools/benchmarks/classification/dist_train_linear.sh configs/benchmarks/classification/imagenet/resnet18_DIV2K_more.py work_dirs/selfsup/simclr/simclr_resnet18_epoch2000_temp0_1_DIV2K/weights_2000.pth work_dirs/selfsup/simclr/transfer_9cls_moco_resnet18_epoch1000_temp0_07_DIV2K/

# test results
bash ./tools/dist_test.sh configs/benchmarks/classification/imagenet/resnet18_DIV2K.py work_dirs/selfsup/InUse/transfer_5cls_moco_resnet18_epoch1000_temp0_07_DIV2K/epoch_100.pth 1

bash ./tools/dist_test.sh configs/benchmarks/classification/imagenet/resnet18_all.py work_dirs/selfsup/InUse/transfer_5cls_moco_resnet18_epoch1000_temp0_07_DIV2K/epoch_100.pth 1

# MOCO supcon esayres
# extract weights
python ./tools/model_converters/extract_backbone_weights.py work_dirs/selfsup/InUse/moco_easyres_epoch2000_temp0_07_DIV2K_supcon/epoch_2000.pth work_dirs/selfsup/InUse/moco_easyres_epoch2000_temp0_07_DIV2K_supcon/weights_2000.pth

# transfer learning
bash ./tools/benchmarks/classification/dist_train_linear.sh configs/benchmarks/classification/imagenet/easyres_all.py work_dirs/selfsup/InUse/moco_easyres_epoch2000_temp0_07_DIV2K_supcon/weights_2000.pth work_dirs/selfsup/InUse/transfer_5cls_moco_easyres_epoch2000_temp0_07_DIV2K_supcon/

# for more cls
bash ./tools/benchmarks/classification/dist_train_linear.sh configs/benchmarks/classification/imagenet/easyres_all.py  work_dirs/selfsup/InUse/moco_easyres_epoch2000_temp0_07_DIV2K_supcon/weights_2000.pth work_dirs/selfsup/InUse/transfer_9cls_moco_easyres_epoch2000_temp0_07_DIV2K_supcon/

# not good, give up
#bash ./tools/benchmarks/classification/dist_train_linear.sh configs/benchmarks/classification/imagenet/easyres_all.py  work_dirs/selfsup/InUse/moco_easyres_epoch2000_temp0_07_DIV2K_supcon/weights_2000.pth work_dirs/selfsup/InUse/transfer_9cls_aniso_moco_easyres_epoch2000_temp0_07_DIV2K_supcon/

# test results
bash ./tools/dist_test.sh configs/benchmarks/classification/imagenet/easyres_all.py work_dirs/selfsup/InUse/transfer_5cls_moco_easyres_epoch2000_temp0_07_DIV2K_supcon/epoch_100.pth 1

bash ./tools/dist_test.sh configs/benchmarks/classification/imagenet/easyres_all.py work_dirs/selfsup/InUse/transfer_9cls_moco_easyres_epoch2000_temp0_07_DIV2K_supcon/epoch_100.pth 1

#bash ./tools/dist_test.sh configs/benchmarks/classification/imagenet/easyres_all.py work_dirs/selfsup/InUse/transfer_9cls_aniso_moco_easyres_epoch2000_temp0_07_DIV2K_supcon/epoch_100.pth 1

# SimCLR supcon esayres
# extract weights
python ./tools/model_converters/extract_backbone_weights.py work_dirs/selfsup/InUse/simclr_easyres_0_07_DIV2K/epoch_2000.pth work_dirs/selfsup/InUse/simclr_easyres_0_07_DIV2K/weights_2000.pth

# transfer learning
bash ./tools/benchmarks/classification/dist_train_linear.sh configs/benchmarks/classification/imagenet/easyres_all.py work_dirs/selfsup/InUse/simclr_easyres_0_07_DIV2K/weights_2000.pth work_dirs/selfsup/InUse/transfer_5cls_simclr_easyres_0_07_DIV2K/


# for more cls
# to be runned


# test results
bash ./tools/dist_test.sh configs/benchmarks/classification/imagenet/easyres_all.py work_dirs/selfsup/InUse/transfer_5cls_simclr_easyres_0_07_DIV2K/epoch_100.pth 1

# Supervised easyres
# train models (not done before)
bash ./tools/dist_train.sh configs/selfsup/simclr/Supervised/simclr_easyres_0_1_cls_all.py 1 --work_dir work_dirs/selfsup/InUse/supervised_easyres_epoch2000_5cls/
# extract weights
python ./tools/model_converters/extract_backbone_weights.py work_dirs/selfsup/InUse/supervised_easyres_epoch2000_5cls/epoch_2000.pth work_dirs/selfsup/InUse/supervised_easyres_epoch2000_5cls/weights_2000.pth

# transfer learning
bash ./tools/benchmarks/classification/dist_train_linear.sh configs/benchmarks/classification/imagenet/easyres_all.py work_dirs/selfsup/InUse/supervised_easyres_epoch2000_5cls/weights_2000.pth work_dirs/selfsup/InUse/transfer_5cls_supervised_easyres_epoch2000/

# for more cls
bash ./tools/benchmarks/classification/dist_train_linear.sh configs/benchmarks/classification/imagenet/easyres_all.py work_dirs/selfsup/InUse/supervised_easyres_epoch2000_5cls/weights_2000.pth work_dirs/selfsup/InUse/transfer_9cls_supervised_easyres_epoch2000/ 

#bash ./tools/benchmarks/classification/dist_train_linear.sh configs/benchmarks/classification/imagenet/easyres_all.py work_dirs/selfsup/InUse/supervised_easyres_epoch2000_5cls/weights_2000.pth work_dirs/selfsup/InUse/transfer_9cls_aniso_supervised_easyres_epoch2000/ 

# test results
bash ./tools/dist_test.sh configs/benchmarks/classification/imagenet/easyres_all.py work_dirs/selfsup/InUse/transfer_5cls_supervised_easyres_epoch2000/epoch_100.pth 1
  # set5 84.0% 
  # set14 91.4% 
  # BSD100 92.6%
  # DIV2K 95.9%
  # Flickr2K 96.1%
  # Urban 93.2%
# 9 cls
bash ./tools/dist_test.sh configs/benchmarks/classification/imagenet/easyres_all.py work_dirs/selfsup/InUse/transfer_9cls_supervised_easyres_epoch2000/epoch_100.pth 1 
# aniso 9 cls  
#bash ./tools/dist_test.sh configs/benchmarks/classification/imagenet/easyres_all.py work_dirs/selfsup/InUse/transfer_9cls_aniso_supervised_easyres_epoch2000/epoch_100.pth 1 

# MOCO supcon esayres our dataset
# X2 data
bash ./tools/dist_train.sh configs/selfsup/moco/mocov2_easyres_Ours_supcon.py 1 --work_dir work_dirs/selfsup/moco/moco_easyres_Ours_supcon_F1/
bash ./tools/dist_train.sh configs/selfsup/moco/mocov2_easyres_Ours_supcon.py 1 --work_dir work_dirs/selfsup/moco/moco_easyres_Ours_supcon_F2/
bash ./tools/dist_train.sh configs/selfsup/moco/mocov2_easyres_Ours_supcon.py 1 --work_dir work_dirs/selfsup/moco/moco_easyres_Ours_supcon_J1/
# X2 sep lens data
bash ./tools/dist_train.sh configs/selfsup/moco/mocov2_easyres_Ours_supcon_X2sep.py 1 --work_dir work_dirs/selfsup/moco/moco_easyres_Ours_supcon_X2sep_S1/

# X4 data 20221130
bash ./tools/dist_train.sh configs/selfsup/moco/mocov2_easyres_Ours_supcon_X4.py 1 --work_dir work_dirs/selfsup/moco/X4_moco_easyres_Ours_supcon_S1/
# X4 normalized
bash ./tools/dist_train.sh configs/selfsup/moco/mocov2_easyres_Ours_supcon_X4norm.py 1 --work_dir work_dirs/selfsup/moco/X4norm_moco_easyres_Ours_supcon_S1/
# X4 with new Cam
bash ./tools/dist_train.sh configs/selfsup/moco/mocov2_easyres_Ours_supcon_X4.py 1 --work_dir work_dirs/selfsup/moco/X4new_moco_easyres_Ours_supcon_S1/

# extract weights
python ./tools/model_converters/extract_backbone_weights.py work_dirs/selfsup/moco/moco_easyres_Ours_supcon/epoch_2000.pth work_dirs/selfsup/moco/moco_easyres_Ours_supcon/weights_2000.pth

python ./tools/model_converters/extract_backbone_weights.py work_dirs/selfsup/moco/moco_easyres_Ours_supcon_F1/epoch_1000.pth work_dirs/selfsup/moco/moco_easyres_Ours_supcon_F1/weights_1000.pth

python ./tools/model_converters/extract_backbone_weights.py work_dirs/selfsup/moco/moco_easyres_Ours_supcon_F2/epoch_1000.pth work_dirs/selfsup/moco/moco_easyres_Ours_supcon_F2/weights_1000.pth

python ./tools/model_converters/extract_backbone_weights.py work_dirs/selfsup/moco/moco_easyres_Ours_supcon_J1/epoch_1000.pth work_dirs/selfsup/moco/moco_easyres_Ours_supcon_J1/weights_1000.pth
# X2 sep lens data
python ./tools/model_converters/extract_backbone_weights.py work_dirs/selfsup/moco/moco_easyres_Ours_supcon_X2sep_S1/epoch_200.pth work_dirs/selfsup/moco/moco_easyres_Ours_supcon_X2sep_S1/weights_200.pth 

# X4
python ./tools/model_converters/extract_backbone_weights.py work_dirs/selfsup/moco/X4_moco_easyres_Ours_supcon_S1/epoch_1000.pth work_dirs/selfsup/moco/X4_moco_easyres_Ours_supcon_S1/weights_1000.pth
# X4 normalized
python ./tools/model_converters/extract_backbone_weights.py work_dirs/selfsup/moco/X4norm_moco_easyres_Ours_supcon_S1/epoch_1000.pth work_dirs/selfsup/moco/X4norm_moco_easyres_Ours_supcon_S1/weights_1000.pth

# X4 newCam
python ./tools/model_converters/extract_backbone_weights.py work_dirs/selfsup/moco/X4new_moco_easyres_Ours_supcon_S1/epoch_1000.pth work_dirs/selfsup/moco/X4new_moco_easyres_Ours_supcon_S1/weights_1000.pth

# transfer learning
bash ./tools/benchmarks/classification/dist_train_linear.sh configs/benchmarks/classification/imagenet/easyres_ours.py work_dirs/selfsup/moco/moco_easyres_Ours_supcon_S1/weights_2000.pth work_dirs/selfsup/moco/transfer_moco_easyres_Ours_supcon_S1/

bash ./tools/benchmarks/classification/dist_train_linear.sh configs/benchmarks/classification/imagenet/easyres_ours.py work_dirs/selfsup/moco/moco_easyres_Ours_supcon_F1/weights_1000.pth work_dirs/selfsup/moco/transfer_moco_easyres_Ours_supcon_F1/

bash ./tools/benchmarks/classification/dist_train_linear.sh configs/benchmarks/classification/imagenet/easyres_ours.py work_dirs/selfsup/moco/moco_easyres_Ours_supcon_F2/weights_1000.pth work_dirs/selfsup/moco/transfer_moco_easyres_Ours_supcon_F2/

bash ./tools/benchmarks/classification/dist_train_linear.sh configs/benchmarks/classification/imagenet/easyres_ours.py work_dirs/selfsup/moco/moco_easyres_Ours_supcon_J1/weights_1000.pth work_dirs/selfsup/moco/transfer_moco_easyres_Ours_supcon_J1/
# X2 sep lens data
bash ./tools/benchmarks/classification/dist_train_linear.sh configs/benchmarks/classification/imagenet/easyres_ours_sep.py work_dirs/selfsup/moco/moco_easyres_Ours_supcon_X2sep_S1/weights_200.pth work_dirs/selfsup/moco/transfer_moco_easyres_Ours_supcon_X2sep_S1/

# X4
bash ./tools/benchmarks/classification/dist_train_linear.sh configs/benchmarks/classification/imagenet/easyres_ours_X4.py work_dirs/selfsup/moco/X4_moco_easyres_Ours_supcon_S1/weights_1000.pth work_dirs/selfsup/moco/transfer_X4_moco_easyres_Ours_supcon_S1/
# X4 normalized
bash ./tools/benchmarks/classification/dist_train_linear.sh configs/benchmarks/classification/imagenet/easyres_ours_X4norm.py work_dirs/selfsup/moco/X4norm_moco_easyres_Ours_supcon_S1/weights_1000.pth work_dirs/selfsup/moco/transfer_X4norm_moco_easyres_Ours_supcon_S1/
# X4 newCam
bash ./tools/benchmarks/classification/dist_train_linear.sh configs/benchmarks/classification/imagenet/easyres_ours_X4new.py work_dirs/selfsup/moco/X4new_moco_easyres_Ours_supcon_S1/weights_1000.pth work_dirs/selfsup/moco/transfer_X4new_moco_easyres_Ours_supcon_S1/

# test results
bash ./tools/dist_test.sh configs/benchmarks/classification/imagenet/easyres_ours.py work_dirs/selfsup/moco/transfer_moco_easyres_Ours_supcon_S1/epoch_100.pth 1

bash ./tools/dist_test.sh configs/benchmarks/classification/imagenet/easyres_ours.py work_dirs/selfsup/moco/transfer_moco_easyres_Ours_supcon_F1/epoch_100.pth 1

bash ./tools/dist_test.sh configs/benchmarks/classification/imagenet/easyres_ours.py work_dirs/selfsup/moco/transfer_moco_easyres_Ours_supcon_F2/epoch_100.pth 1

bash ./tools/dist_test.sh configs/benchmarks/classification/imagenet/easyres_ours.py work_dirs/selfsup/moco/transfer_moco_easyres_Ours_supcon_J1/epoch_100.pth 1
# X2 sep lens
bash ./tools/dist_test.sh configs/benchmarks/classification/imagenet/easyres_ours_sep.py work_dirs/selfsup/moco/transfer_moco_easyres_Ours_supcon_X2sep_S1/epoch_100.pth 1
# X4
bash ./tools/dist_test.sh configs/benchmarks/classification/imagenet/easyres_ours_X4.py work_dirs/selfsup/moco/transfer_X4_moco_easyres_Ours_supcon_S1/epoch_100.pth 1
# X4 normalized
bash ./tools/dist_test.sh configs/benchmarks/classification/imagenet/easyres_ours_X4norm.py work_dirs/selfsup/moco/transfer_X4norm_moco_easyres_Ours_supcon_S1/epoch_100.pth 1

# X4 newCam
bash ./tools/dist_test.sh configs/benchmarks/classification/imagenet/easyres_ours_X4new.py work_dirs/selfsup/moco/transfer_X4new_moco_easyres_Ours_supcon_S1/epoch_100.pth 1

bash ./tools/dist_test.sh configs/benchmarks/classification/imagenet/easyres_ours_X4new.py work_dirs/selfsup/moco/transfer_X4new640_moco_easyres_Ours_supcon_S1/epoch_100.pth 1
# Supervised easyres our dataset
# train models 
bash ./tools/dist_train.sh configs/selfsup/simclr/Supervised/simclr_easyres_0_1_cls3.py 1 --work_dir work_dirs/selfsup/supervised/supervised_easyres_epoch2000_3cls/ 
# extract weights
python ./tools/model_converters/extract_backbone_weights.py work_dirs/selfsup/supervised/supervised_easyres_epoch2000_3cls/epoch_2000.pth work_dirs/selfsup/supervised/supervised_easyres_epoch2000_3cls/weights_2000.pth

# transfer learning
bash ./tools/benchmarks/classification/dist_train_linear.sh configs/benchmarks/classification/imagenet/easyres_ours.py work_dirs/selfsup/supervised/supervised_easyres_epoch2000_3cls/weights_2000.pth work_dirs/selfsup/supervised/transfer_supervised_easyres_epoch2000_3cls/

# for more cls
#bash ./tools/benchmarks/classification/dist_train_linear.sh configs/benchmarks/classification/imagenet/easyres_all.py work_dirs/selfsup/InUse/supervised_easyres_epoch2000_5cls/weights_2000.pth work_dirs/selfsup/InUse/transfer_9cls_supervised_easyres_epoch2000/ 

#bash ./tools/benchmarks/classification/dist_train_linear.sh configs/benchmarks/classification/imagenet/easyres_all.py work_dirs/selfsup/InUse/supervised_easyres_epoch2000_5cls/weights_2000.pth work_dirs/selfsup/InUse/transfer_9cls_aniso_supervised_easyres_epoch2000/ 

# test results
bash ./tools/dist_test.sh configs/benchmarks/classification/imagenet/easyres_ours.py work_dirs/selfsup/supervised/transfer_supervised_easyres_epoch2000_3cls/epoch_100.pth 1



############################################################
# try other methods
# train
	# MOCO resnet18
	# supervised resnet18
	# supervised six-layer
	bash ./tools/dist_train.sh configs/selfsup/simclr/Supervised/simclr_easyres_0_1_cls.py 1 --work_dir work_dirs/selfsup/InUse/supervised_easyres_realMicron/
	bash ./tools/dist_train.sh configs/selfsup/simclr/Supervised/simclr_easyres_0_1_cls.py 1 --work_dir work_dirs/selfsup/InUse/supervised_easyres_realMicron_cls3/
	# simCLR resnet18 superCon
	# simCLR six-layer superCon
	bash ./tools/dist_train.sh configs/selfsup/simclr/Ours/simclr_easyres_0_07_realMicron.py 1 --work_dir work_dirs/selfsup/simclr/simclr_easyres_0_07_realMicron_cls3/
	# get before training weights
	bash ./tools/dist_train.sh configs/selfsup/simclr/Ours/simclr_easyres_0_07_realMicron.py 1 --work_dir work_dirs/selfsup/simclr/simclr_easyres_0_07_realMicron_initial/
	# MoCo resnet18 superCon
	# MoCo six-layer superCon
	bash ./tools/dist_train.sh configs/selfsup/moco/mocov2_easyres_Ours_supcon.py 1 --work_dir work_dirs/selfsup/moco/moco_easyres_realMicron_cls3/
# extract weights
	# supervised six-layer
	python ./tools/model_converters/extract_backbone_weights.py work_dirs/selfsup/InUse/supervised_easyres_realMicron/epoch_300.pth work_dirs/selfsup/InUse/supervised_easyres_realMicron/weights_300.pth
	# simCLR six-layer superCon
	python ./tools/model_converters/extract_backbone_weights.py work_dirs/selfsup/simclr/simclr_easyres_0_07_realMicron/epoch_300.pth work_dirs/selfsup/simclr/simclr_easyres_0_07_realMicron/weights_300.pth 
	# simCLR six-layer superCon cls 3
	python ./tools/model_converters/extract_backbone_weights.py work_dirs/selfsup/simclr/simclr_easyres_0_07_realMicron_cls3/epoch_1000.pth work_dirs/selfsup/simclr/simclr_easyres_0_07_realMicron_cls3/weights_1000.pth 
	# MoCo six-layer superCon
	python ./tools/model_converters/extract_backbone_weights.py work_dirs/selfsup/moco/moco_easyres_realMicron/epoch_300.pth work_dirs/selfsup/moco/moco_easyres_realMicron/weights_300.pth 
	# MoCo six-layer superCon cls 3
	python ./tools/model_converters/extract_backbone_weights.py work_dirs/selfsup/moco/moco_easyres_realMicron_cls3/epoch_1000.pth work_dirs/selfsup/moco/moco_easyres_realMicron_cls3/weights_1000.pth 	
# transfer
	# supervised six-layer
	bash ./tools/benchmarks/classification/dist_train_linear.sh configs/benchmarks/classification/imagenet/easyres_ours4.py work_dirs/selfsup/InUse/supervised_easyres_realMicron/weights_300.pth work_dirs/selfsup/InUse/transfer_supervised_easyres_realMicron/
	
	# simCLR six-layer superCon
	bash ./tools/benchmarks/classification/dist_train_linear.sh configs/benchmarks/classification/imagenet/easyres_ours4.py work_dirs/selfsup/simclr/simclr_easyres_0_07_realMicron/weights_300.pth work_dirs/selfsup/simclr/transfer_simclr_easyres_0_07_realMicron/	
	# simCLR six-layer superCon cls 3
	bash ./tools/benchmarks/classification/dist_train_linear.sh configs/benchmarks/classification/imagenet/easyres_ours.py work_dirs/selfsup/simclr/simclr_easyres_0_07_realMicron_cls3/weights_1000.pth work_dirs/selfsup/simclr/transfer_simclr_easyres_0_07_realMicron_cls3/
	# MoCo six-layer superCon
	bash ./tools/benchmarks/classification/dist_train_linear.sh configs/benchmarks/classification/imagenet/easyres_ours4.py work_dirs/selfsup/moco/moco_easyres_realMicron/weights_300.pth work_dirs/selfsup/moco/transfer_moco_easyres_realMicron/
	# MoCo six-layer superCon cls 3
	bash ./tools/benchmarks/classification/dist_train_linear.sh configs/benchmarks/classification/imagenet/easyres_ours.py work_dirs/selfsup/moco/moco_easyres_realMicron_cls3/weights_1000.pth work_dirs/selfsup/moco/transfer_moco_easyres_realMicron_cls3/	
# test
	# supervised six-layer
	bash ./tools/dist_test.sh configs/benchmarks/classification/imagenet/easyres_ours4.py work_dirs/selfsup/InUse/transfer_supervised_easyres_realMicron/epoch_100.pth 1
	bash ./tools/dist_test.sh configs/benchmarks/classification/imagenet/easyres_ours.py work_dirs/selfsup/InUse/supervised_easyres_realMicron_cls3/epoch_300.pth 1
	# simCLR six-layer superCon
	bash ./tools/dist_test.sh configs/benchmarks/classification/imagenet/easyres_ours4.py work_dirs/selfsup/simclr/transfer_simclr_easyres_0_07_realMicron/epoch_100.pth 1	
	# simCLR cls 3
	bash ./tools/dist_test.sh configs/benchmarks/classification/imagenet/easyres_ours.py work_dirs/selfsup/simclr/transfer_simclr_easyres_0_07_realMicron_cls3/epoch_100.pth 1
	# MoCo six-layer superCon
	bash ./tools/dist_test.sh configs/benchmarks/classification/imagenet/easyres_ours4.py work_dirs/selfsup/moco/transfer_moco_easyres_realMicron/epoch_100.pth 1
	# MoCo cls 3
	bash ./tools/dist_test.sh configs/benchmarks/classification/imagenet/easyres_ours.py work_dirs/selfsup/moco/transfer_moco_easyres_realMicron_cls3/epoch_100.pth 1

##############################################################
# 12192022 visualization
# simclr supercon
python ./tools/analysis_tools/visualize_tsne.py configs/selfsup/simclr/Ours/simclr_easyres_0_07_DIV2K.py --checkpoint work_dirs/selfsup/simclr/simclr_easyres_0_07_DIV2K/epoch_2000.pth --dataset_config configs/benchmarks/classification/tsne_multidevice.py --layer_ind "12" --work_dir work_dirs/selfsup/simclr/tsne_try/

# supervised
python ./tools/analysis_tools/visualize_tsne.py configs/selfsup/simclr/Supervised/simclr_easyres_0_1_cls_all.py --checkpoint work_dirs/selfsup/InUse/supervised_easyres_epoch2000_5cls/epoch_2000.pth --dataset_config configs/benchmarks/classification/tsne_multidevice.py --layer_ind "12" --work_dir work_dirs/selfsup/supervised/tsne_try/

# moco supercon
python ./tools/analysis_tools/visualize_tsne.py configs/selfsup/moco/mocov2_easyres_DIV2K_supcon.py --checkpoint work_dirs/selfsup/moco/moco_easyres_epoch2000_temp0_07_DIV2K_supcon/epoch_2000.pth --dataset_config configs/benchmarks/classification/tsne_multidevice.py --layer_ind "12" --work_dir work_dirs/selfsup/moco/tsne_try/

# simclr supercon before learning
python ./tools/analysis_tools/visualize_tsne.py configs/selfsup/simclr/Ours/simclr_easyres_0_07_DIV2K.py --checkpoint work_dirs/selfsup/simclr/simclr_easyres_0_07_DIV2K_initial/epoch_1.pth --dataset_config configs/benchmarks/classification/tsne_multidevice.py --layer_ind "12" --work_dir work_dirs/selfsup/simclr_inital/tsne_try/

# Real-Micron
# simclr supercon
python ./tools/analysis_tools/visualize_tsne.py configs/selfsup/simclr/Ours/simclr_easyres_0_07_realMicron.py --checkpoint work_dirs/selfsup/simclr/simclr_easyres_0_07_realMicron_cls3/epoch_1000.pth --dataset_config configs/benchmarks/classification/tsne_realMicron.py --layer_ind "1" --work_dir work_dirs/selfsup/simclr_realMicron/tsne_try/

# moco supercon
python ./tools/analysis_tools/visualize_tsne.py configs/selfsup/moco/mocov2_easyres_Ours_supcon.py --checkpoint work_dirs/selfsup/moco/moco_easyres_realMicron_cls3/epoch_1000.pth --dataset_config configs/benchmarks/classification/tsne_realMicron.py --layer_ind "1" --work_dir work_dirs/selfsup/moco_realMicron/tsne_try/

# supervised
python ./tools/analysis_tools/visualize_tsne.py configs/selfsup/simclr/Supervised/simclr_easyres_0_1_cls.py --checkpoint work_dirs/selfsup/InUse/supervised_easyres_realMicron_cls3/epoch_300.pth --dataset_config configs/benchmarks/classification/tsne_realMicron.py --layer_ind "1" --work_dir work_dirs/selfsup/supervised_realMicron/tsne_try/

# before learning
python ./tools/analysis_tools/visualize_tsne.py configs/selfsup/simclr/Ours/simclr_easyres_0_07_realMicron.py --checkpoint work_dirs/selfsup/simclr/simclr_easyres_0_07_realMicron_initial/epoch_1.pth --dataset_config configs/benchmarks/classification/tsne_realMicron.py --layer_ind "1" --work_dir work_dirs/selfsup/simclr_realMicron_initial/tsne_try/

# DRealSR
'''
not good, ImagePairs too many pics
# x2
python ./tools/analysis_tools/visualize_tsne.py configs/selfsup/moco/mocov2_easyres_DRealSR_x2_supcon.py --checkpoint work_dirs/selfsup/moco/moco_easyres_DRealSR_x2/epoch_2000.pth --dataset_config configs/benchmarks/classification/tsne_multidevice.py --layer_ind "12" --work_dir work_dirs/selfsup/moco/tsne_DRealSR_x2/
'''
# x4
python ./tools/analysis_tools/visualize_tsne.py configs/selfsup/moco/mocov2_easyres_DRealSR_supcon.py --checkpoint work_dirs/selfsup/moco/moco_easyres_DRealSR/epoch_2000.pth --dataset_config configs/benchmarks/classification/tsne_multidevice.py --layer_ind "12" --work_dir work_dirs/selfsup/moco/tsne_DRealSR_x4/
# before learning
python ./tools/analysis_tools/visualize_tsne.py configs/selfsup/moco/mocov2_easyres_DRealSR_supcon.py --checkpoint work_dirs/selfsup/moco/moco_easyres_DRealSR_initial/epoch_1.pth --dataset_config configs/benchmarks/classification/tsne_multidevice.py --layer_ind "12" --work_dir work_dirs/selfsup/moco/tsne_DRealSR_x4_initial/
################################################################

##############################################################################
# moco supercon ConvG 12212022
bash ./tools/dist_train.sh configs/selfsup/moco/mocov2_easyres_DIV2K_supcon_ConvG.py 1 --work_dir work_dirs/selfsup/moco/moco_easyres_DIV2K_supcon_ConvG/
##############################################################################

##############################################################################
# begin the DRealSR experiments

# MoCo six-layer superCon
# x4
bash ./tools/dist_train.sh configs/selfsup/moco/mocov2_easyres_DRealSR_supcon.py 1 --work_dir work_dirs/selfsup/moco/moco_easyres_DRealSR/

bash ./tools/dist_train.sh configs/selfsup/moco/mocov2_easyres_DRealSR_supcon.py 1 --work_dir work_dirs/selfsup/moco/moco_easyres_DRealSR_initial/

# simclr
bash ./tools/dist_train.sh configs/selfsup/simclr/Ours/simclr_easyres_DRealSR.py 1 --work_dir work_dirs/selfsup/simclr/simclr_easyres_DRealSR/
# supervised
bash ./tools/dist_train.sh configs/selfsup/simclr/Supervised/simclr_easyres_cls_DRealSR.py 1 --work_dir work_dirs/selfsup/Supervised/supervised_easyres_DRealSR/

# x2
bash ./tools/dist_train.sh configs/selfsup/moco/mocov2_easyres_DRealSR_x2_supcon.py 1 --work_dir work_dirs/selfsup/moco/moco_easyres_DRealSR_x2/

# extract weights
# x4
python ./tools/model_converters/extract_backbone_weights.py work_dirs/selfsup/moco/moco_easyres_DRealSR/epoch_2000.pth work_dirs/selfsup/moco/moco_easyres_DRealSR/weights_2000.pth

python ./tools/model_converters/extract_backbone_weights.py work_dirs/selfsup/simclr/simclr_easyres_DRealSR/epoch_200.pth work_dirs/selfsup/simclr/simclr_easyres_DRealSR/weights_200.pth
# x2
python ./tools/model_converters/extract_backbone_weights.py work_dirs/selfsup/moco/moco_easyres_DRealSR_x2/epoch_2000.pth work_dirs/selfsup/moco/moco_easyres_DRealSR_x2/weights_2000.pth

# transfer learning
bash ./tools/benchmarks/classification/dist_train_linear.sh configs/benchmarks/classification/imagenet/easyres_DRealSR_x4.py work_dirs/selfsup/moco/moco_easyres_DRealSR/weights_2000.pth work_dirs/selfsup/moco/transfer_moco_easyres_DRealSR/

bash ./tools/benchmarks/classification/dist_train_linear.sh configs/benchmarks/classification/imagenet/easyres_DRealSR_x4.py work_dirs/selfsup/simclr/simclr_easyres_DRealSR/weights_200.pth work_dirs/selfsup/simclr/transfer_simclr_easyres_DRealSR/

# test
bash ./tools/dist_test.sh configs/benchmarks/classification/imagenet/easyres_DRealSR_x4.py work_dirs/selfsup/moco/transfer_moco_easyres_DRealSR/epoch_10.pth 1

bash ./tools/dist_test.sh configs/benchmarks/classification/imagenet/easyres_DRealSR_x4.py work_dirs/selfsup/Supervised/supervised_easyres_DRealSR/epoch_200.pth 1

bash ./tools/dist_test.sh configs/benchmarks/classification/imagenet/easyres_DRealSR_x4.py work_dirs/selfsup/simclr/transfer_simclr_easyres_DRealSR/epoch_10.pth 1

'''
Checked, no problem!
# transfer
	# MoCo six-layer superCon
	bash ./tools/benchmarks/classification/dist_train_linear.sh configs/benchmarks/classification/imagenet/easyres_DRealSR.py work_dirs/selfsup/moco/moco_easyres_DRealSR_x2/weights_2000.pth work_dirs/selfsup/moco/transfer_moco_easyres_DRealSR_x2/
'''




