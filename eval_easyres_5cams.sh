# try other methods
# train

	# supervised six-layer
	bash ./tools/dist_train.sh configs/selfsup/simclr/Supervised/simclr_easyres_5cams.py 1 --work_dir work_dirs/selfsup/supervised_easyres_5cams/

	# test supervised six-layer
	bash ./tools/dist_test.sh configs/benchmarks/classification/imagenet/easyres_ours5cams.py work_dirs/selfsup/supervised_easyres_5cams/epoch_300.pth 1





	# simCLR six-layer superCon
	bash ./tools/dist_train.sh configs/selfsup/simclr/Ours/simclr_easyres_0_07_realMicron.py 1 --work_dir work_dirs/selfsup/simclr/simclr_easyres_0_07_realMicron_cls3/
	# get before training weights
	bash ./tools/dist_train.sh configs/selfsup/simclr/Ours/simclr_easyres_0_07_realMicron.py 1 --work_dir work_dirs/selfsup/simclr/simclr_easyres_0_07_realMicron_initial/
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

