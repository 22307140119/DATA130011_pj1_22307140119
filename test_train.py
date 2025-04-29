# An example of read in the data and train the model. The runner is implemented, while the model used for training need your implementation.
import mynn as nn
from draw_tools.plot import plot

import numpy as np
from struct import unpack
import gzip
import matplotlib.pyplot as plt
import pickle

# fixed seed for experiment 随机种子
np.random.seed(309)

# 训练集
train_images_path = r'.\dataset\MNIST\train-images-idx3-ubyte.gz'
train_labels_path = r'.\dataset\MNIST\train-labels-idx1-ubyte.gz'

with gzip.open(train_images_path, 'rb') as f:   # 训练图像
        magic, num, rows, cols = unpack('>4I', f.read(16))
        train_imgs=np.frombuffer(f.read(), dtype=np.uint8).reshape(num, 28*28)
    
with gzip.open(train_labels_path, 'rb') as f:   # 训练标签
        magic, num = unpack('>2I', f.read(8))
        train_labs = np.frombuffer(f.read(), dtype=np.uint8)


# choose 10000 samples from train set as validation set. 随机选验证集
idx = np.random.permutation(np.arange(num))     # 索引随机重排
# save the index.
with open('idx.pickle', 'wb') as f:     # 保存打乱后的索引
        pickle.dump(idx, f)
train_imgs = train_imgs[idx]    # 重排
train_labs = train_labs[idx]
valid_imgs = train_imgs[:10000] # 验证集
valid_labs = train_labs[:10000]
train_imgs = train_imgs[10000:] # 训练集
train_labs = train_labs[10000:]

# normalize from [0, 255] to [0, 1] 数据标准化
train_imgs = train_imgs / train_imgs.max()
valid_imgs = valid_imgs / valid_imgs.max()


# 数据增强的模式
# AUGMENTATION = 'None'
# AUGMENTATION = 'Static'         # 静态数据增强
AUGMENTATION = 'Dynamic'        # 动态数据增强


# 定义模型
# linear_model = nn.models.Model_MLP(train_imgs.shape[-1], [600], 10, 'ReLU', [1e-4, 1e-4])
linear_model = nn.models.Model_MLP(train_imgs.shape[-1], [600], 10, 'ReLU', [1e-4, 1e-4], initialization_method='Kaiming', dropout_p=0.5)
# linear_model = nn.models.Model_MLP(train_imgs.shape[-1], [512,128], 10, 'ReLU', [1e-4, 1e-4, 1e-4])

# 定义优化器
# optimizer = nn.optimizer.SGD(init_lr=0.06, model=linear_model)
optimizer = nn.optimizer.MomentGD(init_lr=0.03, model=linear_model, mu=0.9)

# 定义学习率调度器
scheduler = nn.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=[800, 2400, 4000], gamma=0.5)
# scheduler = nn.lr_scheduler.StepLR(optimizer=optimizer, step_size=400, gamma=0.9)

# 定义损失函数
loss_fn = nn.op.MultiCrossEntropyLoss(model=linear_model, max_classes=train_labs.max()+1)

# 数据增强
if AUGMENTATION == 'None':      # 无数据增强
    data_augmentor = None
elif AUGMENTATION == 'Static':  # 静态数据增强，直接对训练集作用，然后 data_augmentor 变回 None
    data_augmentor = nn.data_augmentation.MyDataAugmentor(max_shift_h=3, max_shift_w=3, max_angle=0.1, scale_range=(0.9, 1.1), prob=0.4, padding=0)
    train_imgs = train_imgs.reshape(train_imgs.shape[0], 1, 28, 28)
    train_imgs = data_augmentor.data_augmentation(train_imgs)
    train_imgs = train_imgs.reshape(train_imgs.shape[0], 784)
    data_augmentor = None
elif AUGMENTATION == 'Dynamic': # 动态数据增强，设置 data_augmentor 供 runner 使用
    data_augmentor = nn.data_augmentation.MyDataAugmentor(max_shift_h=3, max_shift_w=3, max_angle=0.1, scale_range=(0.9, 1.1), prob=0.4, padding=0)


# 定义训练器
runner = nn.runner.RunnerM(linear_model, optimizer, nn.metric.accuracy, loss_fn, scheduler=scheduler, data_augmentor=data_augmentor, image_shape=(1,28,28))


# 启动训练
runner.train([train_imgs, train_labs], [valid_imgs, valid_labs], num_epochs=5, log_iters=100, save_dir=r'./best_models')


# 绘制损失曲线
# _, axes = plt.subplots(1, 2)
# axes.reshape(-1)
# _.set_tight_layout(1)
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
axes = axes.reshape(-1)
fig.set_tight_layout(1)
plot(runner, axes)

plt.show()