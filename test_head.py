from mmselfsup.models.algorithms import SimCLR_Multidevice; 
from mmselfsup.datasets.multi_device import MultiDeviceDataset
from mmselfsup.datasets.data_sources.multidevice import MultiDevice
from mmselfsup.datasets import build_dataloader, build_dataset
from mmselfsup.models.builder import build_algorithm
import torch;
# model = dict(
#     type='MoCo_label',
#     queue_len=8192,
#     feat_dim=128,
#     momentum=0.999,
#     backbone=dict(
#         type='EasyRes',
#         in_channels=3,
#         ),
#     neck=dict(
#         type='MoCoV2Neck',
#         in_channels=512,
#         hid_channels=2048,
#         out_channels=128,
#         with_avg_pool=True),
#     head=dict(type='SNNLossHead', temperature=0.07))
# model = MoCo_label( 
#                     queue_len=8192,
#                     feat_dim=128,
#                     momentum=0.999,
#                     init_cfg = '/home/rui/Rui_SR/mmselfsup/work_dirs/selfsup/moco/mocov2_easyres_DIV2KFlickr2K/epoch_2000.pth',
#                     backbone=dict(
#                     type='EasyRes',
#                     in_channels=3,
#                     ),
#                     neck=dict( 
#                         type='MoCoV2Neck',
#                         in_channels=512,
#                         hid_channels=2048,
#                         out_channels=128,
#                         with_avg_pool=True
#                         ),
#                     head = dict(type='SNNLossHead', temperature=0.07),
#                     )
model = SimCLR_Multidevice(
    # type='SimCLR_Multidevice',
    init_cfg = '/home/rui/Rui_SR/mmselfsup/work_dirs/selfsup/simclr/simclr_resnet18_epoch2000_temp0_07_DIV2K/epoch_2000.pth',
    backbone=dict(
        type='ResNet',
        depth=18,
        in_channels=3,
        out_indices=[4],  # 0: conv-1, x: stage-x
        norm_cfg=dict(type='SyncBN'),
        #frozen_stages= 4
        ),
    neck=dict(
        type='NonLinearNeck',  # SimCLR non-linear neck
        in_channels=512,
        hid_channels=2048,
        out_channels=128,
        num_layers=2,
        with_avg_pool=True),
    head=dict(type='SNNLossHead', temperature=0.1))



model.cuda().eval()

test_pipeline = [
    dict(type='RandomCrop', size=160),
    dict(type='RandomHorizontalFlip'),
    dict(type='RandomVerticalFlip'),
    dict(type='ToTensor'),
]
# data_loaders = [
#     build_dataloader(
#         ds,
#         cfg.data.imgs_per_gpu,
#         cfg.data.workers_per_gpu,
#         # cfg.gpus will be ignored if distributed
#         num_gpus=len(cfg.gpu_ids),
#         dist=distributed,
#         replace=getattr(cfg.data, 'sampling_replace', False),
#         seed=cfg.seed,
#         drop_last=getattr(cfg.data, 'drop_last', False),
#         prefetch=cfg.prefetch,
#         persistent_workers=cfg.persistent_workers,
#         img_norm_cfg=cfg.img_norm_cfg) for ds in dataset
# ]
dataset = MultiDeviceDataset(  
            data_source=dict(
                            type='MultiDevice',
                            ######################### changed to tiny-imagenet ################
                            # data_prefix='data/ThreeDevices', #########################
                            #data_prefix='data/MultiDegrade/SupER1/X4/test', #########################
                            data_prefix='data/DIV2K_Flickr2K/lq/X4', #########################
                            ann_file= None, #######################            
                            # data_prefix='data/imagenet/train', #########################
                            # ann_file='data/imagenet/meta/train.txt', #######################
                            ),
            num_views = [1], 
            pipelines = [test_pipeline],
     )


# test_imgs = dataset.__getitem__(0)
# imgs = []
labels = []
# feats = []
save_folder = '/home/rui/Rui_SR/mmselfsup/data/feats_resnet_neck/'
for i, data_dict in enumerate(dataset):
    # imgs.append(data_dict['img'])
    feat = model.extract_feat(data_dict['img'][0].unsqueeze(0).cuda())
    feat = feat[0].squeeze(-1).squeeze(-1)
    # print(feat.shape,':',i)
    torch.save(feat, save_folder + str(i).zfill(5) + '.pt')
    labels.append(data_dict['label']) 

torch.save(labels, save_folder+'labels.pt')

# test_img = torch.rand(10,3,160,160)
# out_backbone = model.extract_feat(test_img)
# simMatrix = torch.tensor([[1.0, 0.1, 0.8, 0.2, 0.8, 0.1], 
# 		            [0.1, 1.0, 0.3, 0.9, 0.2, 0.75], 
# 		            [0.8, 0.3, 1.0, 0.25, 0.8, 0.2], 
# 		            [0.2, 0.9, 0.25, 1.0, 0.1, 0.9],
# 		            [0.8, 0.2, 0.8, 0.1, 1.0, 0.3],
# 		            [0.1, 0.75, 0.2, 0.9, 0.3, 1.0]
# 		          ])
# 		          
# labels1 = torch.tensor([1,0,1,0,1,0])
# labels2 = torch.tensor([1,1,1,0,1,0])
# net = SNNLossHead(0.10)
# loss1 = net(simMatrix, labels1)
# loss2 = net(simMatrix, labels2)
# print(loss1, loss2)

# net = SNNLossHead(1)
# loss1 = net(simMatrix, labels1)
# loss2 = net(simMatrix, labels2)
# print(loss1, loss2)

# net = SNNLossHead(0.050)
# loss1 = net(simMatrix, labels1)
# loss2 = net(simMatrix, labels2)
# print(loss1, loss2)
# In[]
from sklearn import svm
import torch
import os
import numpy as np
load_folder = '/home/rui/Rui_SR/mmselfsup/data/feats_resnet/'
filenames = sorted(os.listdir(load_folder) )
feats = []
num = 3550

for filename in filenames:
    if 'label' not in filename:
        feat = torch.load(load_folder+filename)
        feat = feat[0].detach().cpu().numpy().reshape(-1)
        feats.append(feat)
feats = np.stack(feats, axis=0)    
labels = torch.load(load_folder+'labels.pt') 
labels = np.stack(labels, axis=0)
clf = svm.SVC(decision_function_shape='ovr')
indices = np.random.permutation(feats.shape[0])
Xtrain = feats[indices[:10000],:]
Xtest = feats[indices[10000:],:]
Ytrain = labels[indices[:10000]]
Ytest = labels[indices[10000:]]
clf.fit(Xtrain, Ytrain)

dec = clf.decision_function(Xtest)
Ypred = np.argmax(dec,axis=1)
print(np.sum(Ytest==Ypred))
# X = [np.array([1,1,1,1]),np.array([0,0,0,1]),np.array([0,0,1,1]),np.array([0,1,1,1])]
# Y = [np.array(4),np.array(1),np.array(2),np.array(3)]

# clf.fit(X,Y)


