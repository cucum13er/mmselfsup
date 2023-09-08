# try other methods
# train

	# supervised six-layer
	bash ./tools/dist_train.sh configs/selfsup/simclr/Supervised/simclr_fcdd_0_1_cls.py 1 --work_dir work_dirs/selfsup/InUse/supervised_fcdd_realMicron/
	# supervised six-layer sepTarget
	bash ./tools/dist_train.sh configs/selfsup/simclr/Supervised/simclr_fcdd_cls_sepTarget.py 1 --work_dir work_dirs/selfsup/InUse/supervised_fcdd_realMicron_sepTarget/	
	
		# test
		bash ./tools/dist_test.sh configs/benchmarks/classification/imagenet/fcdd_ours.py work_dirs/selfsup/InUse/supervised_fcdd_realMicron/epoch_300.pth 1
		bash ./tools/dist_test.sh configs/benchmarks/classification/imagenet/fcdd_ours.py work_dirs/selfsup/InUse/supervised_fcdd_realMicron_sepTarget/epoch_300.pth 1
########### not run yet ####################################################################################
	bash ./tools/dist_train.sh configs/selfsup/simclr/Supervised/simclr_easyres_0_1_cls.py 1 --work_dir work_dirs/selfsup/InUse/supervised_easyres_realMicron_cls3/
	# simCLR resnet18 superCon
	# simCLR six-layer superCon
	bash ./tools/dist_train.sh configs/selfsup/simclr/Ours/simclr_easyres_0_07_realMicron.py 1 --work_dir work_dirs/selfsup/simclr/simclr_easyres_0_07_realMicron_cls3/
	# get before training weights
	bash ./tools/dist_train.sh configs/selfsup/simclr/Ours/simclr_easyres_0_07_realMicron.py 1 --work_dir work_dirs/selfsup/simclr/simclr_easyres_0_07_realMicron_initial/
	# MoCo resnet18 superCon
	# MoCo six-layer superCon
	bash ./tools/dist_train.sh configs/selfsup/moco/mocov2_easyres_Ours_supcon.py 1 --work_dir work_dirs/selfsup/moco/moco_easyres_realMicron_cls3/
	
####################################################################################################################################################################################	
