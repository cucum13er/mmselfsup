#bash ./tools/dist_train.sh configs/selfsup/simclr/simclr_resnet50_8xb32-coslr-200e_in1k_Rui.py 1 --work_dir work_dirs/selfsup/simclr/simclr_resnet50_epoch10000_temp0.1/

#bash ./tools/dist_train.sh configs/selfsup/simclr/simclr_resnet50_8xb32-coslr-200e_in1k_Rui0_01.py 1 --work_dir work_dirs/selfsup/simclr/simclr_resnet50_epoch10000_temp0.01/

#bash ./tools/dist_train.sh configs/selfsup/simclr/simclr_resnet50_8xb32-coslr-200e_in1k_Rui01.py 1 --work_dir work_dirs/selfsup/simclr/simclr_resnet50_epoch10000_temp01/

#bash ./tools/dist_train.sh configs/selfsup/simclr/simclr_resnet50_8xb32-coslr-200e_in1k_Rui10.py 1 --work_dir work_dirs/selfsup/simclr/simclr_resnet50_epoch10000_temp10/


# bash ./tools/dist_train.sh configs/selfsup/simclr/simclr_resnet18_0_1.py 1 --work_dir work_dirs/selfsup/simclr/simclr_resnet18_epoch1000_temp0_1_DIV2K/

#bash ./tools/dist_train.sh configs/selfsup/simclr/simclr_resnet18_0_01.py 1 --work_dir work_dirs/selfsup/simclr/simclr_resnet18_epoch3000_temp0_01/

#bash ./tools/dist_train.sh configs/selfsup/simclr/simclr_resnet18_1_0.py 1 --work_dir work_dirs/selfsup/simclr/simclr_resnet18_epoch3000_temp1_0/

#bash ./tools/dist_train.sh configs/selfsup/simclr/simclr_resnet18_10.py 1 --work_dir work_dirs/selfsup/simclr/simclr_resnet18_epoch3000_temp10/

#bash ./tools/dist_train.sh configs/selfsup/simclr/simclr_resnet18_0_1.py 1 --work_dir work_dirs/selfsup/simclr/simclr_resnet18_epoch100_temp0_1_withPre/ 

# bash ./tools/dist_train.sh configs/selfsup/simclr/simclr_resnet18_0_1_nolabel.py 1 --work_dir work_dirs/selfsup/simclr/simclr_resnet18_epoch1000_temp0_1_nolabel/

# bash ./tools/dist_train.sh configs/selfsup/simclr/simclr_resnet18_0_1_cls.py 1 --work_dir work_dirs/selfsup/simclr/simclr_resnet18_epoch1000_temp0_1_cls/

#################### without normalization################################
# nolabel
#bash ./tools/dist_train.sh configs/selfsup/moco/mocov2_resnet18_DIV2K.py 1 --work_dir work_dirs/selfsup/moco/moco_resnet18_epoch2000_temp0_07_DIV2K/
# bash ./tools/dist_train.sh configs/selfsup/moco/mocov2_resnet18_Flickr2K.py 1 --work_dir work_dirs/selfsup/moco/moco_resnet18_epoch2000_temp0_07_Flickr2K/
# bash ./tools/dist_train.sh configs/selfsup/moco/mocov2_resnet18_SupER.py 1 --work_dir work_dirs/selfsup/moco/moco_resnet18_epoch2000_temp0_07_SupER/

#bash ./tools/dist_train.sh configs/selfsup/simclr/Nolabel/simclr_resnet18_0_1_nolabel_DIV2K.py 1 --work_dir work_dirs/selfsup/simclr/simclr_resnet18_epoch2000_temp0_1_DIV2K_nolabel/
# bash ./tools/dist_train.sh configs/selfsup/simclr/Nolabel/simclr_resnet18_0_1_nolabel_SupER.py 1 --work_dir work_dirs/selfsup/simclr/simclr_resnet18_epoch2000_temp0_1_SupER_nolabel/
# bash ./tools/dist_train.sh configs/selfsup/simclr/Nolabel/simclr_resnet18_0_1_nolabel_Flickr2K.py 1 --work_dir work_dirs/selfsup/simclr/simclr_resnet18_epoch2000_temp0_1_Flickr2K_nolabel/

# ours
#bash ./tools/dist_train.sh configs/selfsup/simclr/Ours/simclr_resnet18_0_1_DIV2K.py 1 --work_dir work_dirs/selfsup/simclr/simclr_resnet18_epoch2000_temp0_1_DIV2K/
# bash ./tools/dist_train.sh configs/selfsup/simclr/Ours/simclr_resnet18_0_1_Flickr2K.py 1 --work_dir work_dirs/selfsup/simclr/simclr_resnet18_epoch2000_temp0_1_Flickr2K/
#bash ./tools/dist_train.sh configs/selfsup/simclr/Ours/simclr_resnet18_0_1_SupER.py 1 --work_dir work_dirs/selfsup/simclr/simclr_resnet18_epoch2000_temp0_07_SupER/

# 04282022 new
#bash ./tools/dist_train.sh configs/selfsup/moco/mocov2_resnet18_DIV2K_supcon.py 1 --work_dir work_dirs/selfsup/moco/moco_resnet18_epoch2000_temp0_07_DIV2K_supcon/

#bash ./tools/dist_train.sh configs/selfsup/moco/mocov2_easyres_DIV2K_supcon.py 1 --work_dir work_dirs/selfsup/moco/moco_easyres_epoch2000_temp0_07_DIV2K_supcon/

bash ./tools/dist_train.sh configs/selfsup/simclr/Ours/simclr_easyres_0_07_DIV2K.py 1 --work_dir work_dirs/selfsup/simclr/simclr_easyres_0_07_DIV2K_initial/

# anisotropic
#bash ./tools/dist_train.sh configs/selfsup/simclr/Ours/simclr_resnet18_0_1_DIV2K.py 1 --work_dir work_dirs/selfsup/simclr/simclr_resnet18_epoch2000_temp0_1_DIV2K_aniso/


# fine tune temperature
# bash ./tools/dist_train.sh configs/selfsup/simclr/Ours/simclr_resnet18_0_07_DIV2K.py 1 --work_dir work_dirs/selfsup/simclr/simclr_resnet18_epoch2000_temp0_07_DIV2K/
#bash ./tools/dist_train.sh configs/selfsup/simclr/Ours/simclr_resnet18_0_05_DIV2K.py 1 --work_dir work_dirs/selfsup/simclr/simclr_resnet18_epoch2000_temp0_05_DIV2K/
#bash ./tools/dist_train.sh configs/selfsup/simclr/Ours/simclr_resnet18_0_03_DIV2K.py 1 --work_dir work_dirs/selfsup/simclr/simclr_resnet18_epoch2000_temp0_03_DIV2K/


# supervised
# DIV2K cls
#bash ./tools/dist_train.sh configs/selfsup/simclr/Supervised/simclr_resnet18_0_1_cls_DIV2K.py 1 --work_dir work_dirs/selfsup/simclr/simclr_resnet18_epoch2000_temp0_1_cls/
# Flickr2K cls
#bash ./tools/dist_train.sh configs/selfsup/simclr/Supervised/simclr_resnet18_0_1_cls_Flickr2K.py 1 --work_dir work_dirs/selfsup/simclr/simclr_resnet18_epoch2000_temp0_1_Flickr2K_cls/
# SupER cls
#bash ./tools/dist_train.sh configs/selfsup/simclr/Supervised/simclr_resnet18_0_1_cls_SupER.py 1 --work_dir work_dirs/selfsup/simclr/simclr_resnet18_epoch2000_temp0_1_SupER_cls/



# evaluate EasyRes

# train DIV2K_Flickr2K
#bash ./tools/dist_train.sh configs/selfsup/moco/mocov2_easyres_DIV2KFlickr2K.py 1 --work_dir work_dirs/selfsup/moco/mocov2_easyres_DIV2KFlickr2K/


# our dataset
# train
bash ./tools/dist_train.sh configs/selfsup/moco/mocov2_easyres_Ours_supcon.py 1 --work_dir work_dirs/selfsup/moco/moco_easyres_Ours_supcon/
# 




