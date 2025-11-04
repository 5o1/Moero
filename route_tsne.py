"""
Step2. T-SNE可视化
使用sklearn中的TSNE进一步降维到2维或3维, 并可视化
"""

import os
import torch
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from typing import List

# 根据文件名分组打标签(要求文件名格式为 tag1@tag2@tag3@... )
def make_group(files, group_by:int | List[int] = 0):
    group = {}
    for i, filename in enumerate(files):
        tagchain = files[i].split("_")
        if isinstance(group_by, list):
            group_key = "_".join([tagchain[j] for j in group_by])
        else:
            group_key = files[i].split("_")[group_by]
        group.setdefault(group_key, []).append(i)
    return group

# 设置特征文件路径
path = "route_ml.npz"  # step.1 输出的文件
fat = "/home/lyy/dataset/brain-knee/val-fat.txt"
fatlist = []
with open(fat, 'r', encoding='utf-8') as f:
    fat_lines = f.readlines()
    for line in fat_lines:
        fname, flag = line.strip().split()
        if int(flag) == 1:
            fatlist.append(fname)

output_dim = 2  # 可视化维度, 2或3

# 加载特征(step.1 输出的文件)
print("Loading features...")
with np.load(path, allow_pickle=True) as data:
    features = data["features"]
    filenames = data["fnames"]
    filenames = [os.path.basename(f) for f in filenames]

print(f"Loaded {len(features)} feature files.")

# normalize features
# features = np.array(features)
# features = (features - features.mean(axis=0, keepdims=True)) / (features.std(axis=0, keepdims=True) + 1e-8)

for i, f in enumerate(filenames):
    if "AX" in f:
        continue
    if f in fatlist:
        f = "file_knee_knee-fatsup_unknown_" + f
    else:
        f = "file_knee_knee_unknown_" + f
    filenames[i] = f

group = make_group(filenames, group_by=[2])
group = {key: group[key] for key in sorted(group)}
print("group by:", group.keys())

# 在每个group选择n_select个样本进行绘图
# selected_indices = []
# n_select = 100
# for key, indices in group.items():
#     if len(indices) <= n_select:
#         selected_indices.extend(indices)
#     else:
#         selected_indices.extend(np.random.choice(indices, size=n_select, replace=False).tolist())
# features = features[selected_indices]
# filenames = [filenames[i] for i in selected_indices]
# group = make_group(filenames, group_by=[2])

# shuffle
random_indices = np.random.choice(len(features), size=len(features), replace=False)
features = features[random_indices]
filenames = [filenames[i] for i in random_indices]

# # 只需要文件名中不包含AX的样本
indices = []
for i, f in enumerate(filenames):
    if "AX" in f:
        indices.append(i)

features = features[indices]
filenames = [filenames[i] for i in indices]
group = make_group(filenames, group_by=[2])


# 检查数据维度
print(f"Feature shape: {features.shape}")

# TSNE feature reduce to 2
print("Performing TSNE...")
tsne = TSNE(n_components=2, perplexity=15, learning_rate=10)
features_embedded = tsne.fit_transform(features) # embedding到2维
print("AfterEmbedded shape:", features_embedded.shape)

# 只需要文件名中有AX的样本
# indices = []
# for i, f in enumerate(filenames):
#     if "AX" in f:
#         indices.append(i)

# features_embedded = features_embedded[indices]
# filenames = [filenames[i] for i in indices]
# group = make_group(filenames, group_by=[2])

print("After filter, group by:", group.keys())
for key in group:
    print(f"{key}: {len(group[key])} samples")


# plot
def plot_tsne(embedded_features, filenames, output_dim=2):
    plt.figure(figsize=(10, 8))
    cmap = plt.colormaps.get_cmap("tab20") # 找一个好看的cmap
    colors = {groupname: cmap(i/len(group)) for i, groupname in enumerate(group.keys())}
    print(colors)

    if output_dim == 2:
        for i, (groupname, indice) in enumerate(group.items()):
            plt.scatter(embedded_features[indice, 0], embedded_features[indice, 1], alpha=1, s=50, color = colors[groupname], label = groupname)
        plt.legend(loc='upper left', bbox_to_anchor=(1.05, 1))  # 图例放在右侧外部

        plt.title("t-SNE Visualization (2D)")
        plt.xlabel("t-SNE Dimension 1")
        plt.ylabel("t-SNE Dimension 2")
        plt.tight_layout()
    elif output_dim == 3:
        from mpl_toolkits.mplot3d import Axes3D
        ax = plt.figure().add_subplot(111, projection='3d')
        for i, (groupname, indice) in enumerate(group.items()):
            plt.scatter(embedded_features[indice, 0], embedded_features[indice, 1], embedded_features[indice, 2], alpha=1, s=50, color = colors[groupname], label = groupname)
        plt.legend(loc='upper left', bbox_to_anchor=(1.05, 1))  # 图例放在右侧外部
        ax.set_title("t-SNE Visualization (3D)")
        ax.set_xlabel("t-SNE Dimension 1")
        ax.set_ylabel("t-SNE Dimension 2")
        ax.set_zlabel("t-SNE Dimension 3")
        plt.tight_layout()
    else:
        raise ValueError("Invalid output dimension. Only 2D or 3D are supported.")
    # plt.show()

# 绘制结果
plot_tsne(features_embedded, filenames, output_dim=output_dim)

# 保存图片
from datetime import datetime
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

plt.savefig(f"tsne_fuse_visualization_{timestamp}.png")