# dataset settings
data_source = 'MultiDevice_ours'
dataset_type = 'MultiDeviceDataset_ours'
img_norm_cfg = dict(mean=[0.485, ], std=[0.229,])
# The difference between mocov2 and mocov1 is the transforms in the pipeline
train_pipeline = [
    # dict(type='CenterCrop', size=)
    dict(type='RandomCrop', size=120),
    dict(type='RandomHorizontalFlip'),
    dict(type='RandomVerticalFlip'),
]
test_pipeline = [
    # dict(type='', size=)
    dict(type='RandomCrop', size=120),
    dict(type='RandomHorizontalFlip'),
    dict(type='RandomVerticalFlip'),
    # dict(type='ToTensor'),
]
# prefetch
prefetch = False
# if not prefetch:
#     train_pipeline.extend(
#         [dict(type='ToTensor'), # scale to [0,1]
#          #dict(type='Normalize', **img_norm_cfg)
#          ])

# dataset summary
data = dict(
    imgs_per_gpu=128,  # total 32*8=256
    workers_per_gpu=8,
    drop_last=True,
    train=dict(
        type=dataset_type,
        data_source=dict(
            type=data_source,
            ######################### changed to tiny-imagenet ################
            # data_prefix='data/ThreeDevices', #########################
            data_prefix='data/Ours_contrastive_withNew/X4_train/', #########################
            #data_prefix='data/DIV2K_Flickr2K/lq/X4', #########################
            ann_file= None, #######################
            # data_prefix='data/imagenet/train', #########################
            # ann_file='data/imagenet/meta/train.txt', #######################
        ),
        num_views=[2],
        pipelines=[train_pipeline],
        prefetch=prefetch,
    ),
    val=dict(
        type=dataset_type,
        data_source=dict(
            type=data_source,
            ######################### changed to tiny-imagenet ################
            # data_prefix='data/ThreeDevices', #########################
            #data_prefix='data/MultiDegrade/SupER1/X4/val', #########################
            data_prefix='data/Ours_contrastive_withNew/X4_test/', #########################
            ann_file= None, #######################            
            # data_prefix='data/imagenet/train', #########################
            # ann_file='data/imagenet/meta/train.txt', #######################
        ),
        num_views=[1],
        pipelines=[test_pipeline],
        prefetch=prefetch,        
        ),
    test=dict(
        type=dataset_type,
        data_source=dict(
            type=data_source,
            ######################### changed to tiny-imagenet ################
            # data_prefix='data/ThreeDevices', #########################
            #data_prefix='data/MultiDegrade/SupER1/X4/test', #########################
            data_prefix='data/Ours_contrastive_withNew/X4_test/', #########################
            ann_file= None, #######################            
            # data_prefix='data/imagenet/train', #########################
            # ann_file='data/imagenet/meta/train.txt', #######################
        ),
        num_views=[1],
        pipelines=[test_pipeline],
        prefetch=prefetch,
        ),
    
    )


