# Copyright (c) OpenMMLab. All rights reserved.
from .cae_vit import CAEViT
from .mae_vit import MAEViT
from .maskfeat_vit import MaskFeatViT
from .mim_cls_vit import MIMVisionTransformer
from .resnet import ResNet, ResNetV1d
from .resnext import ResNeXt
from .simmim_swin import SimMIMSwinTransformer
from .vision_transformer import VisionTransformer
###################################################################
from .easyres import EasyRes
from .easyres_ConvG import EasyRes_ConvG
from .fcdd_rui import FCDD_Rui
__all__ = [
    'ResNet', 'ResNetV1d', 'ResNeXt', 'MAEViT', 'MIMVisionTransformer',
    'VisionTransformer', 'SimMIMSwinTransformer', 'CAEViT', 'MaskFeatViT','EasyRes', 'EasyRes_ConvG', 'FCDD_Rui'
]
