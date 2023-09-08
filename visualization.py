import numpy as np
import torch.nn as nn
from sklearn.manifold import TSNE
import pickle

device = 'cuda:0'



a_file = open("/home/rui/Rui_SR/mmselfsup/work_dirs/selfsup/simclr_resnet18_0_1_DIV2K/features_neck.pkl", "rb")
output = pickle.load(a_file)

features = output['features']
labels = output['labels']


f = []
for feature in features:
    if feature.ndim > 1:
        tmp = feature.mean(axis=(1,2))
    else:
        tmp = feature
    f.append(tmp)
    
    
# f = np.array(features)
# f = np.concatenate(features, 0)
f_min = np.min(f, 0)
f_max = np.max(f, 0)

# normalization
f_norm = (f - f_min) / (f_max - f_min)

# T-SNE
tsne = TSNE(n_components=2, init='pca', random_state=0)
embed = tsne.fit_transform(f)
embed = embed.reshape(5, -1, 2)

# visualization
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

plt.figure(figsize=(5, 5))
ax = plt.subplot(111)
# plt.scatter(embed[0, :, 0], embed[0, :, 1], embed[0, :, 2], c='b')
# plt.scatter(embed[1, :, 0], embed[1, :, 1], embed[1, :, 2], c='r')
# plt.scatter(embed[2, :, 0], embed[2, :, 1], embed[2, :, 2], c='g')
# plt.scatter(embed[3, :, 0], embed[3, :, 1], embed[3, :, 2], c='k')
# plt.scatter(embed[4, :, 0], embed[4, :, 1], embed[4, :, 2], c='c')
# plt.xticks(fontsize=14)
# plt.yticks(fontsize=14)
# # plt.zticks(fontsize=14)
# plt.show()
plt.scatter(embed[0, :, 0], embed[0, :, 1], c='b')
plt.scatter(embed[1, :, 0], embed[1, :, 1], c='r')
plt.scatter(embed[2, :, 0], embed[2, :, 1], c='g')
plt.scatter(embed[3, :, 0], embed[3, :, 1], c='k')
plt.scatter(embed[4, :, 0], embed[4, :, 1], c='c')
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
# plt.zticks(fontsize=14)
plt.show()









# # degradation settings
# args.scale = [4]
# args.blur_kernel = 21
# args.blur_type = 'aniso_gaussian'
# lambda_1_list = [0.5, 4.0, 2.0, 3.2]
# lambda_2_list = [0.5, 4.0, 1.0, 1.5]
# theta_list    = [0,   0,   30,  135]
# noise = 0

# # paths
# pth_path = 'blindsr_x'+str(args.scale[0])+'_aniso'
# img_path = 'F:/LongguangWang/Data/benchmark/B100/HR/*.png'


# if __name__ == '__main__':
#     net = BlindSR(args).to(device)
#     net.load_state_dict(torch.load('experiment/blindsr_x4_bicubic_aniso/model/model_600.pt'), strict=False)
#     net.eval()

#     HR_img_list = glob.glob(img_path)
#     fea_list = []

#     for lambda_1, lambda_2, theta in zip(lambda_1_list, lambda_2_list, theta_list):
#         degrade = util.SRMDPreprocessing(
#             scale=args.scale[0],
#             kernel_size=args.blur_kernel,
#             blur_type=args.blur_type,
#             lambda_1=lambda_1,
#             lambda_2=lambda_2,
#             theta=theta,
#             noise=noise
#         )

#         with torch.no_grad():
#             for i in range(len(HR_img_list)):
#                 # read HR images
#                 HR_img = imageio.imread(HR_img_list[i])
#                 if np.ndim(HR_img) < 3:
#                     HR_img = np.stack([HR_img, HR_img, HR_img], 2)
#                 HR_img = np.ascontiguousarray(HR_img.transpose((2, 0, 1)))
#                 HR_img = torch.from_numpy(HR_img).float().to(device).unsqueeze(0).unsqueeze(1)
#                 b, n, c, h, w = HR_img.size()
#                 HR_img = HR_img[:, :, :, :h // args.scale[0] * args.scale[0], :w // args.scale[0] * args.scale[0]]

#                 # generate LR images
#                 LR_img, _ = degrade(HR_img, random=False)
#                 LR_img = LR_img.to(device)

#                 # generate degradation representations
#                 _, fea = net.E.encoder_q(LR_img[:, 0, ...])
#                 fea_list.append(fea.data.cpu().numpy())

#     f = np.concatenate(fea_list, 0)
#     f_min = np.min(f, 0)
#     f_max = np.max(f, 0)

#     # normalization
#     f_norm = (f - f_min) / (f_max - f_min)

#     # T-SNE
#     tsne = TSNE(n_components=2, init='pca', random_state=0)
#     embed = tsne.fit_transform(f)
#     embed = embed.reshape(len(lambda_1_list), 1, 100, -1)

#     # visualization
#     import matplotlib
#     matplotlib.use('TkAgg')
#     import matplotlib.pyplot as plt

#     plt.figure(figsize=(5, 5))
#     ax = plt.subplot(111)
#     plt.scatter(embed[0, 0, :, 0], embed[0, 0, :, 1], c='b')
#     plt.scatter(embed[1, 0, :, 0], embed[1, 0, :, 1], c='r')
#     plt.scatter(embed[2, 0, :, 0], embed[2, 0, :, 1], c='g')
#     plt.scatter(embed[3, 0, :, 0], embed[3, 0, :, 1], c='k')
#     plt.xticks(fontsize=14)
#     plt.yticks(fontsize=14)
#     plt.show()
