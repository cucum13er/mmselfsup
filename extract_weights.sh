#python ./tools/model_converters/extract_backbone_weights.py work_dirs/selfsup/simclr/simclr_resnet50_8xb32-coslr-200e_in1k_5G/epoch_1000.pth work_dirs/selfsup/simclr/simclr_resnet50_8xb32-coslr-200e_in1k_5G/simclr_1000epoch_5G.pth

bash tools/benchmarks/classification/svm_voc07/tools/eval_svm_full_Rui.sh work_dirs/selfsup/simclr/simclr_resnet50_8xb32-coslr-200e_in1k_5G

#/home/rui/Rui_SR/mmselfsup/configs/selfsup/simclr/simclr_resnet50_8xb32-coslr-200e_in1k_Rui.py 1 /home/rui/Rui_SR/mmselfsup/work_dirs/selfsup/simclr/simclr_resnet50_8xb32-coslr-200e_in1k_5G/epoch_1000.pth

#python tools/benchmarks/classification/svm_voc07/tools/train_svm_kfold_parallel.py --data_file work_dirs/selfsup/simclr/simclr_resnet50_8xb32-coslr-200e_in1k_5G/features/voc07_trainval_feat5.npy --targets_data_file data/MultiDegrade/SupER1/X4/train_labels.npy --costs_list 1.0,10.0,100.0 --output_path work_dirs/selfsup/simclr/simclr_resnet50_8xb32-coslr-200e_in1k_5G/svm/voc07_feat5

