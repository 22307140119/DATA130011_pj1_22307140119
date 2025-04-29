# An example of read in the data and train the model. The runner is implemented, while the model used for training need your implementation.
import mynn as nn
from draw_tools.plot import plot_cnn

import numpy as np
from struct import unpack
import gzip
import matplotlib.pyplot as plt
import pickle
import time

# fixed seed for experiment 随机种子
np.random.seed(3)

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
num_valid = 10000
valid_imgs = train_imgs[:num_valid] # 验证集
valid_labs = train_labs[:num_valid]
train_imgs = train_imgs[num_valid:] # 训练集
train_labs = train_labs[num_valid:]

# 调试用
# MODE = 'debug'
MODE = 'train'

# 数据增强的模式
# AUGMENTATION = 'None'
# AUGMENTATION = 'Static'         # 静态数据增强
AUGMENTATION = 'Dynamic'        # 动态数据增强

if MODE == 'debug':
    n = 380
    valid_imgs = train_imgs[:20] # 验证集
    valid_labs = train_labs[:20]
    train_imgs = train_imgs[20:n] # 训练集
    train_labs = train_labs[20:n]
    logiters = 1
    numepochs = 4
elif MODE == 'train':
    logiters = 100
    numepochs = 4



# normalize from [0, 255] to [0, 1] 数据标准化
train_imgs = train_imgs / train_imgs.max()
valid_imgs = valid_imgs / valid_imgs.max()



# 调整数据形状，原本是 N*784，调整成 N*1*28*28 
# MNIST 数据集是“灰度”图像，只有一个通道
N_train = train_imgs.shape[0]
train_imgs = train_imgs.reshape(N_train, 1, 28, 28)
N_valid = valid_imgs.shape[0]
valid_imgs = valid_imgs.reshape(N_valid, 1, 28, 28)



# 初始化模型
cnn_model = nn.models.Model_CNN(
        layers = [
        # bs*1*28*28 --> bs*32*28*28
        nn.op.conv2D(in_channels=1, out_channels=32, kernel_size=3, padding=1),
        nn.op.ReLU(),
        # bs*32*28*28 --> bs*32*14*14
        nn.op.MaxPooling(2, 2),
        
        # bs*32*14*14 --> bs*64*14*14
        nn.op.conv2D(in_channels=32, out_channels=64, kernel_size=3, padding=1),
        nn.op.ReLU(),
        # bs*64*14*14 --> bs*64*7*7
        nn.op.MaxPooling(2, 2),
        
        # 两个全连接层
        nn.op.Flatten(),
        nn.op.Linear(64*7*7, 512, initialize_method='Kaiming'),
        nn.op.ReLU(),
        nn.op.Linear(512, 10, initialize_method='Kaiming')
        ],
        dropout_p=0.5
    )

# 定义优化器（学习率1e-3疑似有点大，但是也能跑）
# optimizer = nn.optimizer.SGD(init_lr=1e-3, model=cnn_model)
optimizer = nn.optimizer.MomentGD(init_lr=1e-3, model=cnn_model, mu=0.9)

# 学习率调度策略
scheduler = nn.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=[800, 2400, 3200, 4800], gamma=0.2)

# 损失函数
loss_fn = nn.op.MultiCrossEntropyLoss(model=cnn_model, max_classes=10)

# 数据增强
if AUGMENTATION == 'None':      # 无数据增强
    data_augmentor = None
elif AUGMENTATION == 'Static':  # 静态数据增强，直接对训练集作用，然后 data_augmentor 变回 None
    data_augmentor = nn.data_augmentation.MyDataAugmentor(max_shift_h=3, max_shift_w=3, max_angle=0.1, scale_range=(0.9, 1.1), prob=0.5, padding=0)
    train_imgs = data_augmentor.data_augmentation(train_imgs)
    data_augmentor = None
elif AUGMENTATION == 'Dynamic': # 动态数据增强，设置 data_augmentor 供 runner 使用
    data_augmentor = nn.data_augmentation.MyDataAugmentor(max_shift_h=3, max_shift_w=3, max_angle=0.1, scale_range=(0.9, 1.1), prob=0.5, padding=0)


# 训练器
runner = nn.runner.RunnerM(cnn_model, optimizer, nn.metric.accuracy, loss_fn, scheduler=scheduler, data_augmentor=data_augmentor, image_shape=(1,28,28))
print('Runner set up')

start_time = time.time() 
# 启动训练
runner.train([train_imgs, train_labs], 
            [valid_imgs, valid_labs], 
            num_epochs=numepochs,
            log_iters=logiters, 
            save_dir=r'./cnn_models')
end_time = time.time()
print(f'Total time spent: {end_time - start_time}s')

# _, axes = plt.subplots(1, 2)
# axes.reshape(-1)
# _.set_tight_layout(1)
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
axes = axes.reshape(-1)
fig.set_tight_layout(1)
plot_cnn(runner, axes)
plt.show()