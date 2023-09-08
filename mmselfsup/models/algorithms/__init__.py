# Copyright (c) OpenMMLab. All rights reserved.
from .barlowtwins import BarlowTwins
from .base import BaseModel
from .byol import BYOL
from .cae import CAE
from .classification import Classification
from .deepcluster import DeepCluster
from .densecl import DenseCL
from .interclr_moco import InterCLRMoCo
from .mae import MAE
from .maskfeat import MaskFeat
from .mmcls_classifier_wrapper import MMClsImageClassifierWrapper
from .moco import MoCo
from .mocov3 import MoCoV3
from .npid import NPID
from .odc import ODC
from .relative_loc import RelativeLoc
from .rotation_pred import RotationPred
from .simclr import SimCLR
from .simmim import SimMIM
from .simsiam import SimSiam
from .swav import SwAV
##################################################################################################
from .simclr_multidevice import SimCLR_Multidevice
from .simclr_multidevice_cls import SimCLR_Multidevice_cls
from .simclr_multidevice_nolabel import SimCLR_Nolabel
from .moco_label import MoCo_label
__all__ = [
    'BaseModel', 'BarlowTwins', 'BYOL', 'Classification', 'DeepCluster',
    'DenseCL', 'InterCLRMoCo', 'MoCo', 'NPID', 'ODC', 'RelativeLoc',
    'RotationPred', 'SimCLR', 'SimSiam', 'SwAV', 'MAE', 'MoCoV3', 'SimMIM',
    'MMClsImageClassifierWrapper', 'CAE', 'MaskFeat','SimCLR_Multidevice', 'SimCLR_Multidevice_cls', 'SimCLR_Nolabel',
    'MoCo_label'
]
