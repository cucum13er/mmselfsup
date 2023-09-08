# bash ./tools/dist_train.sh configs/selfsup/densecl/densecl_resnet50_8xb32-coslr-200e_in1k.py 1 --work_dir work_dirs/selfsup/densecl/densecl_resnet50_8xb32-coslr-200e_in1k/


#bash ./tools/dist_train.sh configs/selfsup/moco/mocov2_resnet50_8xb32-coslr-200e_in1k_Rui.py 1 --work_dir work_dirs/selfsup/moco/mocov2_resnet50_8xb32-coslr-200e_in1k_Rui/

bash ./tools/dist_train.sh configs/selfsup/simclr/simclr_resnet50_8xb32-coslr-200e_in1k_Rui.py 1 --work_dir work_dirs/selfsup/simclr/simclr_resnet50_8xb32-coslr-200e_in1k_initial/

# bash ./tools/dist_test.sh configs/selfsup/simclr/simclr_resnet50_test.py  /home/rui/Rui_SR/mmselfsup/work_dirs/selfsup/simclr/simclr_resnet50_epoch10000_temp0.1/epoch_10000.pth 1 
# bash ./tools/dist_test.sh configs/selfsup/simclr/simclr_resnet50_8xb32-coslr-200e_in1k_Rui.py  /home/rui/Rui_SR/mmselfsup/work_dirs/selfsup/simclr/simclr_resnet50_8xb32-coslr-200e_in1k_cls/epoch_400.pth 1 
bash ./tools/dist_test.sh configs/selfsup/simclr/simclr_resnet18_test.py  /home/rui/Rui_SR/mmselfsup/work_dirs/selfsup/simclr/simclr_resnet18_epoch1000_temp0_1_randomcrop/epoch_1000.pth 1 

# 2.18.2022 test visualization
bash ./tools/dist_test.sh configs/selfsup/simclr/Ours/simclr_resnet18_0_1_DIV2K.py work_dirs/selfsup/simclr/simclr_resnet18_epoch2000_temp0_1_DIV2K_aniso/epoch_2000.pth 1 


bash ./tools/dist_train.sh configs/selfsup/simclr/simclr_resnet50_8xb32-coslr-200e_in1k_cls.py 1 --work_dir work_dirs/selfsup/simclr/simclr_resnet50_8xb32-coslr-200e_in1k_cls/

bash ./tools/benchmarks/classification/dist_train_linear.sh configs/benchmarks/classification/imagenet/resnet50_8xb32-coslr-100e_in1k_Rui.py work_dirs/selfsup/simclr/simclr_resnet18_8xb32-coslr-200e_in1k_tem0.1/epoch_3000.pth

bash ./tools/benchmarks/classification/dist_train_linear.sh configs/benchmarks/classification/imagenet/resnet18.py work_dirs/selfsup/simclr/simclr_resnet18_epoch1000_temp0_1_randomcrop_initial/epoch_10.pth

bash ./tools/benchmarks/classification/dist_train_linear.sh configs/benchmarks/classification/imagenet/resnet18.py work_dirs/selfsup/simclr/simclr_resnet18_epoch1000_temp0_1_randomcrop/weights_1000.pth
#simclr_resnet18_epoch1000_temp0_1_randomcrop_initial

#/home/rui/Rui_SR/mmselfsup/work_dirs/selfsup/simclr/simclr_resnet18_epoch3000_temp0_1/

python ./tools/model_converters/extract_backbone_weights.py work_dirs/selfsup/simclr/simclr_resnet18_epoch1000_temp0_1_randomcrop/epoch_1000.pth work_dirs/selfsup/simclr/simclr_resnet18_epoch1000_temp0_1_randomcrop/weights_1000.pth

python ./tools/model_converters/extract_all_weights.py work_dirs/benchmarks/classification/imagenet/resnet18/weights_1000.pth/epoch_100.pth work_dirs/benchmarks/classification/imagenet/resnet18/weights_1000.pth/weights_100.pth


python ./tools/model_converters/extract_backbone_weights.py work_dirs/selfsup/simclr/simclr_resnet18_epoch1000_temp0_1_nolabel/epoch_1000.pth work_dirs/selfsup/simclr/simclr_resnet18_epoch1000_temp0_1_nolabel/weights_1000.pth

python ./tools/model_converters/extract_backbone_weights.py work_dirs/selfsup/simclr/simclr_resnet18_epoch1000_temp0_1_cls/epoch_1000.pth work_dirs/selfsup/simclr/simclr_resnet18_epoch1000_temp0_1_cls/weights_1000.pth

bash ./tools/benchmarks/classification/dist_train_linear.sh configs/benchmarks/classification/imagenet/resnet18.py work_dirs/selfsup/simclr/simclr_resnet18_epoch1000_temp0_1_cls/weights_1000.pth

bash ./tools/dist_test.sh configs/benchmarks/classification/imagenet/resnet18.py /home/rui/Rui_SR/mmselfsup/work_dirs/benchmarks/classification/imagenet/resnet18/weights_1000.pth/epoch_100.pth 1

bash ./tools/dist_test.sh configs/benchmarks/classification/imagenet/resnet18.py /home/rui/Rui_SR/mmselfsup/work_dirs/benchmarks/classification/imagenet/resnet18/weights_1000.pth/weights_100.pth 1


########################### formal test begins ############################
# the supervised method test on DIV2K
bash ./tools/dist_test.sh configs/selfsup/simclr/simclr_resnet18_0_1_cls.py  /home/rui/Rui_SR/mmselfsup/work_dirs/selfsup/simclr/simclr_resnet18_epoch1000_temp0_1_DIV2K_cls/epoch_1000.pth 1 
bash ./tools/dist_test.sh configs/selfsup/simclr/Supervised/simclr_resnet18_0_1_cls_DIV2K.py  /home/rui/Rui_SR/mmselfsup/work_dirs/selfsup/simclr/simclr_resnet18_epoch2000_temp0_1_DIV2K_cls/epoch_2000.pth 1 
bash ./tools/dist_test.sh configs/selfsup/simclr/Supervised/simclr_resnet18_0_1_cls_Flickr2K.py  /home/rui/Rui_SR/mmselfsup/work_dirs/selfsup/simclr/simclr_resnet18_epoch2000_temp0_1_DIV2K_cls/epoch_2000.pth 1 
# 04262022
bash ./tools/dist_test.sh configs/selfsup/simclr/Supervised/simclr_resnet18_0_1_cls_all.py  /home/rui/Rui_SR/mmselfsup/work_dirs/selfsup/simclr/simclr_resnet18_epoch2000_temp0_1_DIV2K_cls/epoch_2000.pth 1 


# the supervised method test on DIV2K more cls
## transfer learning from cls5 to cls9
python ./tools/model_converters/extract_backbone_weights.py work_dirs/selfsup/simclr/simclr_resnet18_epoch2000_temp0_1_DIV2K_cls/epoch_2000.pth work_dirs/selfsup/simclr/simclr_resnet18_epoch2000_temp0_1_DIV2K_cls/weights_2000.pth

bash ./tools/benchmarks/classification/dist_train_linear.sh configs/benchmarks/classification/imagenet/resnet18_DIV2K_more.py work_dirs/selfsup/simclr/simclr_resnet18_epoch2000_temp0_1_DIV2K_cls/weights_2000.pth work_dirs/selfsup/simclr/transfer_5cls_9cls_DIV2K_epoch2000/

bash ./tools/dist_test.sh configs/benchmarks/classification/imagenet/resnet18_DIV2K_more.py work_dirs/selfsup/simclr/transfer_5cls_9cls_DIV2K_epoch2000/epoch_100.pth 1

# the supervised method test on SupER
#bash ./tools/dist_test.sh configs/selfsup/simclr/simclr_resnet18_0_1_cls.py  /home/rui/Rui_SR/mmselfsup/work_dirs/selfsup/simclr/#simclr_resnet18_epoch1000_temp0_1_SupER_cls/epoch_1000.pth 1 
#bash ./tools/dist_test.sh configs/selfsup/simclr/Supervised/simclr_resnet18_0_1_cls_SupER.py  /home/rui/Rui_SR/mmselfsup/work_dirs/selfsup/#simclr/simclr_resnet18_epoch2000_temp0_1_SupER_cls/epoch_2000.pth 1 

# the supervised method test on Flickr2K
bash ./tools/dist_test.sh configs/selfsup/simclr/simclr_resnet18_0_1_cls.py  /home/rui/Rui_SR/mmselfsup/work_dirs/selfsup/simclr/simclr_resnet18_epoch1000_temp0_1_Flickr2K_cls/epoch_1000.pth 1 

# transfer learning of contrastive methods
# DIV2K
# get backbone weights
python ./tools/model_converters/extract_backbone_weights.py work_dirs/selfsup/simclr/simclr_resnet18_epoch2000_temp0_1_DIV2K/epoch_2000.pth work_dirs/selfsup/simclr/simclr_resnet18_epoch2000_temp0_1_DIV2K/weights_2000.pth

python ./tools/model_converters/extract_backbone_weights.py work_dirs/selfsup/simclr/simclr_resnet18_epoch2000_temp0_07_DIV2K/epoch_2000.pth work_dirs/selfsup/simclr/simclr_resnet18_epoch2000_temp0_07_DIV2K/weights_2000.pth

python ./tools/model_converters/extract_backbone_weights.py work_dirs/selfsup/simclr/simclr_resnet18_epoch2000_temp0_05_DIV2K/epoch_2000.pth work_dirs/selfsup/simclr/simclr_resnet18_epoch2000_temp0_05_DIV2K/weights_2000.pth

python ./tools/model_converters/extract_backbone_weights.py work_dirs/selfsup/simclr/simclr_resnet18_epoch2000_temp0_03_DIV2K/epoch_2000.pth work_dirs/selfsup/simclr/simclr_resnet18_epoch2000_temp0_03_DIV2K/weights_2000.pth

# 04272022
python ./tools/model_converters/extract_backbone_weights.py work_dirs/selfsup/moco/moco_resnet18_epoch2000_temp0_07_DIV2K_supcon/epoch_2000.pth work_dirs/selfsup/moco/moco_resnet18_epoch2000_temp0_07_DIV2K_supcon/weights_2000.pth

python ./tools/model_converters/extract_backbone_weights.py work_dirs/selfsup/moco/moco_resnet18_epoch2000_temp0_07_DIV2K_supcon_initial/epoch_1.pth work_dirs/selfsup/moco/moco_resnet18_epoch2000_temp0_07_DIV2K_supcon_initial/weights_1.pth

python ./tools/model_converters/extract_backbone_weights.py work_dirs/selfsup/moco/moco_resnet18_epoch2000_temp0_07_DIV2K_supcon/epoch_2000.pth work_dirs/selfsup/moco/moco_resnet18_epoch2000_temp0_07_DIV2K_supcon/weights_2000.pth

python ./tools/model_converters/extract_backbone_weights.py work_dirs/selfsup/moco/mocov2_easyres_DIV2KFlickr2K/epoch_2000.pth work_dirs/selfsup/moco/mocov2_easyres_DIV2KFlickr2K/weights_2000.pth

# 05092022
python ./tools/model_converters/extract_backbone_weights.py work_dirs/selfsup/moco/moco_easyres_epoch2000_temp0_07_DIV2K_supcon/epoch_2000.pth work_dirs/selfsup/moco/moco_easyres_epoch2000_temp0_07_DIV2K_supcon/weights_2000.pth 

# transfer learning
bash ./tools/benchmarks/classification/dist_train_linear.sh configs/benchmarks/classification/imagenet/resnet18_DIV2K.py work_dirs/selfsup/simclr/simclr_resnet18_epoch2000_temp0_1_DIV2K/weights_2000.pth work_dirs/selfsup/simclr/transfer_ours_5cls_DIV2K_epoch2000/

bash ./tools/benchmarks/classification/dist_train_linear.sh configs/benchmarks/classification/imagenet/resnet18_DIV2K.py work_dirs/selfsup/simclr/simclr_resnet18_epoch2000_temp0_07_DIV2K/weights_2000.pth work_dirs/selfsup/simclr/transfer_ours_5cls_DIV2K_epoch2000_temp007/

bash ./tools/benchmarks/classification/dist_train_linear.sh configs/benchmarks/classification/imagenet/resnet18_DIV2K.py work_dirs/selfsup/simclr/simclr_resnet18_epoch2000_temp0_05_DIV2K/weights_2000.pth work_dirs/selfsup/simclr/transfer_ours_5cls_DIV2K_epoch2000_temp005/

bash ./tools/benchmarks/classification/dist_train_linear.sh configs/benchmarks/classification/imagenet/resnet18_DIV2K.py work_dirs/selfsup/simclr/simclr_resnet18_epoch2000_temp0_03_DIV2K/weights_2000.pth work_dirs/selfsup/simclr/transfer_ours_5cls_DIV2K_epoch2000_temp003/

#04272022
bash ./tools/benchmarks/classification/dist_train_linear.sh configs/benchmarks/classification/imagenet/resnet18_DIV2K.py work_dirs/selfsup/moco/moco_resnet18_epoch2000_temp0_07_DIV2K_supcon/weights_2000.pth work_dirs/selfsup/moco/transfer_ours_5cls_DIV2K_epoch2000_temp007_moco/

bash ./tools/benchmarks/classification/dist_train_linear.sh configs/benchmarks/classification/imagenet/resnet18_DIV2K.py work_dirs/selfsup/moco/moco_resnet18_epoch2000_temp0_07_DIV2K_supcon_initial/weights_1.pth work_dirs/selfsup/moco/transfer_ours_5cls_DIV2K_epoch2000_temp007_moco_cmp/

bash ./tools/benchmarks/classification/dist_train_linear.sh configs/benchmarks/classification/imagenet/easyres_all.py work_dirs/selfsup/moco/mocov2_easyres_DIV2KFlickr2K/weights_2000.pth work_dirs/selfsup/moco/transfer_ours_5cls_all_epoch2000_temp007_moco/


# for more cls
bash ./tools/benchmarks/classification/dist_train_linear.sh configs/benchmarks/classification/imagenet/resnet18_DIV2K_more.py work_dirs/selfsup/simclr/simclr_resnet18_epoch2000_temp0_1_DIV2K/weights_2000.pth work_dirs/selfsup/simclr/transfer_ours_9cls_DIV2K_epoch2000/

bash ./tools/benchmarks/classification/dist_train_linear.sh configs/benchmarks/classification/imagenet/resnet18_DIV2K_more.py work_dirs/selfsup/simclr/simclr_resnet18_epoch2000_temp0_07_DIV2K/weights_2000.pth work_dirs/selfsup/simclr/transfer_ours_9cls_DIV2K_epoch2000_temp007/

bash ./tools/benchmarks/classification/dist_train_linear.sh configs/benchmarks/classification/imagenet/resnet18_DIV2K_more.py work_dirs/selfsup/simclr/simclr_resnet18_epoch2000_temp0_05_DIV2K/weights_2000.pth work_dirs/selfsup/simclr/transfer_ours_9cls_DIV2K_epoch2000_temp005/

bash ./tools/benchmarks/classification/dist_train_linear.sh configs/benchmarks/classification/imagenet/resnet18_DIV2K_more.py work_dirs/selfsup/simclr/simclr_resnet18_epoch2000_temp0_03_DIV2K/weights_2000.pth work_dirs/selfsup/simclr/transfer_ours_9cls_DIV2K_epoch2000_temp003/

# test final model
bash ./tools/dist_test.sh configs/benchmarks/classification/imagenet/resnet18_DIV2K.py work_dirs/selfsup/simclr/transfer_ours_5cls_DIV2K_epoch2000/epoch_100.pth 1

bash ./tools/dist_test.sh configs/benchmarks/classification/imagenet/resnet18_DIV2K.py work_dirs/selfsup/simclr/transfer_ours_5cls_DIV2K_epoch2000_temp007/epoch_100.pth 1
# 0.07 temp for cls5: 87.9%
bash ./tools/dist_test.sh configs/benchmarks/classification/imagenet/resnet18_DIV2K.py work_dirs/selfsup/simclr/transfer_ours_5cls_DIV2K_epoch2000_temp005/epoch_100.pth 1
bash ./tools/dist_test.sh configs/benchmarks/classification/imagenet/resnet18_DIV2K.py work_dirs/selfsup/simclr/transfer_ours_5cls_DIV2K_epoch2000_temp003/epoch_100.pth 1

# test final more cls
bash ./tools/dist_test.sh configs/benchmarks/classification/imagenet/resnet18_DIV2K_more.py work_dirs/selfsup/simclr/transfer_ours_9cls_DIV2K_epoch2000/epoch_100.pth 1

bash ./tools/dist_test.sh configs/benchmarks/classification/imagenet/resnet18_DIV2K_more.py work_dirs/selfsup/simclr/transfer_ours_9cls_DIV2K_epoch2000_temp007/epoch_100.pth 1
bash ./tools/dist_test.sh configs/benchmarks/classification/imagenet/resnet18_DIV2K_more.py work_dirs/selfsup/simclr/transfer_ours_9cls_DIV2K_epoch2000_temp005/epoch_100.pth 1

bash ./tools/dist_test.sh configs/benchmarks/classification/imagenet/resnet18_DIV2K_more.py work_dirs/selfsup/simclr/transfer_ours_9cls_DIV2K_epoch2000_temp003/epoch_100.pth 1
# DIV2K No label
# get backbone weights
python ./tools/model_converters/extract_backbone_weights.py work_dirs/selfsup/simclr/simclr_resnet18_epoch2000_temp0_1_DIV2K_nolabel/epoch_2000.pth work_dirs/selfsup/simclr/simclr_resnet18_epoch2000_temp0_1_DIV2K_nolabel/weights_2000.pth

python ./tools/model_converters/extract_backbone_weights.py work_dirs/selfsup/moco/moco_resnet18_epoch2000_temp0_07_DIV2K/epoch_2000.pth work_dirs/selfsup/moco/moco_resnet18_epoch2000_temp0_07_DIV2K/weights_2000.pth

# transfer learning
bash ./tools/benchmarks/classification/dist_train_linear.sh configs/benchmarks/classification/imagenet/resnet18_DIV2K.py work_dirs/selfsup/simclr/simclr_resnet18_epoch2000_temp0_1_DIV2K_nolabel/weights_2000.pth work_dirs/selfsup/simclr/transfer_ours_5cls_DIV2K_epoch2000_nolabel/

bash ./tools/benchmarks/classification/dist_train_linear.sh configs/benchmarks/classification/imagenet/resnet18_DIV2K.py work_dirs/selfsup/simclr/simclr_resnet18_epoch2000_temp0_1_DIV2K_nolabel/weights_2000.pth work_dirs/selfsup/moco/transfer_moco_5cls_DIV2K_epoch2000_temp007/

# test final model
bash ./tools/dist_test.sh configs/benchmarks/classification/imagenet/resnet18_DIV2K.py work_dirs/selfsup/simclr/transfer_ours_5cls_DIV2K_epoch2000_nolabel/epoch_100.pth 1

bash ./tools/dist_test.sh configs/benchmarks/classification/imagenet/resnet18_DIV2K.py work_dirs/selfsup/moco/transfer_moco_5cls_DIV2K_epoch2000_temp007/epoch_100.pth 1


# Flickr2K
# get backbone weights
python ./tools/model_converters/extract_backbone_weights.py work_dirs/selfsup/simclr/simclr_resnet18_epoch1000_temp0_1_Flickr2K/epoch_1000.pth work_dirs/selfsup/simclr/simclr_resnet18_epoch1000_temp0_1_Flickr2K/weights_1000.pth

# transfer learning
bash ./tools/benchmarks/classification/dist_train_linear.sh configs/benchmarks/classification/imagenet/resnet18.py work_dirs/selfsup/simclr/simclr_resnet18_epoch1000_temp0_1_Flickr2K/weights_1000.pth work_dirs/selfsup/simclr/transfer_ours_5cls_Flickr2K/
# 04262022
bash ./tools/benchmarks/classification/dist_train_linear.sh configs/benchmarks/classification/imagenet/resnet18_DIV2K.py work_dirs/selfsup/simclr/simclr_resnet18_epoch2000_temp0_1_DIV2K/weights_2000.pth work_dirs/selfsup/simclr/transfer_ours_5cls_Flickr2K/

# test final model
bash ./tools/dist_test.sh configs/benchmarks/classification/imagenet/resnet18.py work_dirs/selfsup/simclr/transfer_ours_5cls_Flickr2K/epoch_100.pth 1
# 04262022
bash ./tools/dist_test.sh configs/benchmarks/classification/imagenet/resnet18_Flickr2K.py work_dirs/selfsup/simclr/transfer_ours_5cls_Flickr2K/epoch_100.pth 1

bash ./tools/dist_test.sh configs/benchmarks/classification/imagenet/resnet18_all.py work_dirs/selfsup/simclr/transfer_ours_5cls_DIV2K_epoch2000_temp007/epoch_100.pth 1

bash ./tools/dist_test.sh configs/benchmarks/classification/imagenet/resnet18_all.py work_dirs/selfsup/moco/transfer_ours_5cls_DIV2K_epoch2000_temp007_moco/epoch_100.pth 1

bash ./tools/dist_test.sh configs/benchmarks/classification/imagenet/easyres_all.py work_dirs/selfsup/moco/transfer_ours_5cls_all_epoch2000_temp007_moco/epoch_100.pth 1

'
# SupER
# get backbone weights
python ./tools/model_converters/extract_backbone_weights.py work_dirs/selfsup/simclr/simclr_resnet18_epoch1000_temp0_1_SupER/epoch_1000.pth work_dirs/selfsup/simclr/simclr_resnet18_epoch1000_temp0_1_SupER/weights_1000.pth
python ./tools/model_converters/extract_backbone_weights.py work_dirs/selfsup/simclr/simclr_resnet18_epoch2000_temp0_07_SupER/epoch_2000.pth work_dirs/selfsup/simclr/simclr_resnet18_epoch2000_temp0_07_SupER/weights_2000.pth
# transfer learning
bash ./tools/benchmarks/classification/dist_train_linear.sh configs/benchmarks/classification/imagenet/resnet18_SupER.py work_dirs/selfsup/simclr/simclr_resnet18_epoch2000_temp0_07_SupER/weights_2000.pth work_dirs/selfsup/simclr/transfer_ours_5cls_SupER_epoch2000_temp007/

# transfer for more cls
bash ./tools/benchmarks/classification/dist_train_linear.sh configs/benchmarks/classification/imagenet/resnet18_SupER_more.py work_dirs/selfsup/simclr/simclr_resnet18_epoch2000_temp0_07_SupER/weights_2000.pth work_dirs/selfsup/simclr/transfer_ours_9cls_SupER_epoch2000_temp007/

# test final model
bash ./tools/dist_test.sh configs/benchmarks/classification/imagenet/resnet18_SupER.py work_dirs/selfsup/simclr/transfer_ours_5cls_SupER_epoch2000_temp007/epoch_100.pth 1

bash ./tools/dist_test.sh configs/benchmarks/classification/imagenet/resnet18_SupER_more.py work_dirs/selfsup/simclr/transfer_ours_9cls_SupER_epoch2000_temp007/epoch_100.pth 1
'


# new easyres backbone
python ./tools/model_converters/extract_backbone_weights.py work_dirs/selfsup/moco/mocov2_easyres_DIV2KFlickr2K/epoch_2000.pth work_dirs/selfsup/moco/mocov2_easyres_DIV2KFlickr2K/weights_2000.pth

#bash tools/benchmarks/classification/svm_voc07/dist_test_svm_epoch.sh ${SELFSUP_CONFIG} ${EPOCH} ${FEATURE_LIST}




