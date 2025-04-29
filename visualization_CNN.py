# codes to make visualization of your weights.
import mynn as nn
import random
import matplotlib.pyplot as plt

model = nn.models.Model_CNN()
model.load_model(r'.\cnn_models\best_model_9720.pickle')

# 各层参数可视化
LAYER0 = 0
LAYER1 = 0
LAYER2 = 0
LAYER3 = 0

# 图像中间结果可视化
IMAGE = 1
IMAGE0 = 0
IMAGE2 = 0
IMAGE3 = 0
IMAGE5 = 0

# 权重
c0 = model.layers[0].params['W']    # (32, 1, 3, 3)
c1 = model.layers[3].params['W']    # (64, 32, 3, 3)
m2 = model.layers[7].params['W']    # (3136, 512)
m3 = model.layers[9].params['W']    # (512, 10)


if LAYER0:
    # 第一层卷积层
    B, C, H, W = c0.shape
    # print(B, C, H, W)
    fig, axes = plt.subplots(1, 32, figsize=(15, 2))
    for i in range(32):
        f = c0[i].reshape(3, 3)
        im = axes[i].imshow(
            f,
            interpolation='nearest',  # 无插值
        )
        axes[i].axis('off')
        # 图片编号
        axes[i].text(
            2, -1, str(i+1),         # 位置在网格(2,0)处（右下角）
            ha='center', va='center',
            color='gray'
        )

    # 添加 colorbar
    fig.colorbar(im, ax=axes.ravel().tolist(), orientation='horizontal', pad=0.2, aspect=40)
    plt.show()


if LAYER1:
    # 第二层卷积层，理论上有 64*32 ，这里只可视化 16*32
    fig, axes = plt.subplots(16, 32, figsize=(15, 8))
    # 遍历所有子图
    for i in range(16):
        for j in range(32):
            
            f = c1[i, j].reshape(3, 3)
            im = axes[i, j].imshow(
                f,
                # cmap='bwr',      # 颜色映射
                interpolation='nearest',  # 无插值
            )
            axes[i, j].axis('off')
    # 调整子图间距
    plt.subplots_adjust(wspace=0.1, hspace=0.1)
    # 添加 colorbar
    fig.colorbar(im, ax=axes.ravel().tolist(), orientation='horizontal', pad=0.1, aspect=40)
    plt.show()


if LAYER2:
    # 直接可视化第三层全连接层
    plt.figure(figsize=(15, 6))
    plt.imshow(
        m2.T,
        cmap='bwr',      # 颜色映射
        interpolation='nearest',  # 无插值
        # vmin=-0.1,
        # vmax=0.1
    )
    plt.colorbar(label='Value', orientation='horizontal', pad=0.2, aspect=40)
    plt.show()

    # 变成正方形，原 3136 = 64*7*7
    fig, axes = plt.subplots(4, 16, figsize=(15, 6))
    # 遍历所有子图
    for i in range(4):
        for j in range(16):
            idx = 49 * (i*16 + j)
            f = m2[idx:idx+49][:,0].reshape(7, 7)
            im = axes[i, j].imshow(
                f,
                # cmap='bwr',      # 颜色映射
                interpolation='nearest',  # 无插值
            )
            axes[i, j].axis('off')
    # 调整子图间距
    plt.subplots_adjust(wspace=0.1, hspace=0.1)
    # 添加 colorbar
    fig.colorbar(im, ax=axes.ravel().tolist(), orientation='horizontal', pad=0.1, aspect=40)
    plt.show()


if LAYER3:
    # 直接可视化第四层全连接层
    plt.figure(figsize=(15, 2))
    plt.imshow(
        m3.T,
        cmap='bwr',      # 颜色映射
        interpolation='nearest',  # 无插值
        # vmin=-0.8,
        # vmax=0.8
    )
    plt.colorbar(label='Value', orientation='horizontal', pad=0.3, aspect=40)
    plt.show()


if IMAGE:
    import gzip
    from struct import unpack
    import numpy as np

    train_images_path = r'.\dataset\MNIST\train-images-idx3-ubyte.gz'

    with gzip.open(train_images_path, 'rb') as f:   # 训练图像
            magic, num, rows, cols = unpack('>4I', f.read(16))
            train_imgs=np.frombuffer(f.read(), dtype=np.uint8).reshape(num, 28*28)
    
    img = train_imgs[9]
    # 画一下原本的数字
    plt.figure(figsize=(5, 5))
    plt.imshow(img.reshape(28, 28))
    plt.axis('off')  # 关闭坐标轴
    plt.show()


    # 第一层
    layer = model.layers[0]
    res = layer.forward(img.reshape(1,1,28,28)) # (1, 32, 28, 28)
    if IMAGE0:
        # print(res.shape)
        # plt.figure(figsize=(15, 7))
        # for i in range(32):
        #     plt.subplot(4, 8, i+1)
        #     plt.imshow(res[0,i].reshape(28, 28))
        #     plt.axis('off')  # 关闭坐标轴
        # plt.tight_layout()  # 调整子图间距
        # plt.show()

        vmin, vmax = res.min(), res.max()
        # 创建带有预留空间的画布
        fig = plt.figure(figsize=(15, 8))  # 增加高度预留colorbar空间
        im = None  # 用于存储最后一个有效的imshow对象
        # 绘制子图
        for i in range(32):
            ax = plt.subplot(4, 8, i+1)
            im = ax.imshow(res[0,i].reshape(28, 28),
                        vmin=vmin,
                        vmax=vmax,
                        cmap='viridis')
            ax.axis('off')

        # 调整布局并添加全局colorbar
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.12)  # 底部预留15%空间给colorbar
        # 创建colorbar专用axes（左下角坐标，宽度，高度）
        cbar_ax = fig.add_axes([0.15, 0.08, 0.7, 0.03])  # [左，下，宽，高]
        fig.colorbar(im, 
                    cax=cbar_ax, 
                    orientation='horizontal')
        plt.show()


    # 第三层
    res = model.layers[1].forward(res)
    res = model.layers[2].forward(res)
    if IMAGE2:
        # plt.figure(figsize=(15, 7))
        # for i in range(32):
        #     plt.subplot(4, 8, i+1)
        #     plt.imshow(res[0,i].reshape(14, 14))
        #     plt.axis('off')  # 关闭坐标轴
        # plt.tight_layout()  # 调整子图间距
        # plt.show()

        vmin, vmax = res.min(), res.max()/2
        # 创建带有预留空间的画布
        fig = plt.figure(figsize=(15, 8))  # 增加高度预留colorbar空间
        im = None  # 用于存储最后一个有效的imshow对象
        # 绘制子图
        for i in range(32):
            ax = plt.subplot(4, 8, i+1)
            im = ax.imshow(res[0,i].reshape(14, 14),
                        vmin=vmin,
                        vmax=vmax,
                        cmap='viridis')
            ax.axis('off')

        # 调整布局并添加全局colorbar
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.12)  # 底部预留15%空间给colorbar
        # 创建colorbar专用axes（左下角坐标，宽度，高度）
        cbar_ax = fig.add_axes([0.15, 0.08, 0.7, 0.03])  # [左，下，宽，高]
        fig.colorbar(im, 
                    cax=cbar_ax, 
                    orientation='horizontal')
        plt.show()


    # 第四层
    res = model.layers[3].forward(res)  # (1, 64, 14, 14)
    if IMAGE3:
        # print(res.shape)
        # plt.figure(figsize=(15, 4))
        # for i in range(64):
        #     plt.subplot(4, 16, i+1)
        #     plt.imshow(res[0,i].reshape(14, 14))
        #     plt.axis('off')  # 关闭坐标轴
        # plt.tight_layout()  # 调整子图间距
        # plt.show()

        vmin, vmax = res.min(), res.max()/1.5
        # 创建带有预留空间的画布
        fig = plt.figure(figsize=(15, 4))  # 增加高度预留colorbar空间
        im = None  # 用于存储最后一个有效的imshow对象
        # 绘制子图
        for i in range(64):
            ax = plt.subplot(4, 16, i+1)
            im = ax.imshow(res[0,i].reshape(14, 14),
                        vmin=vmin,
                        vmax=vmax,
                        cmap='viridis')
            ax.axis('off')

        # 调整布局并添加全局colorbar
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.12)  # 底部预留15%空间给colorbar
        # 创建colorbar专用axes（左下角坐标，宽度，高度）
        cbar_ax = fig.add_axes([0.15, 0.08, 0.7, 0.03])  # [左，下，宽，高]
        fig.colorbar(im, 
                    cax=cbar_ax, 
                    orientation='horizontal')
        plt.show()


    # 第六层
    res = model.layers[4].forward(res)
    res = model.layers[5].forward(res)  # (1, 64, 7, 7)
    if IMAGE5:
        vmin, vmax = res.min(), res.max()/2
        # 创建带有预留空间的画布
        fig = plt.figure(figsize=(15, 4))  # 增加高度预留colorbar空间
        im = None  # 用于存储最后一个有效的imshow对象
        # 绘制子图
        for i in range(64):
            ax = plt.subplot(4, 16, i+1)
            im = ax.imshow(res[0,i].reshape(7, 7),
                        vmin=vmin,
                        vmax=vmax,
                        cmap='viridis')
            ax.axis('off')

        # 调整布局并添加全局colorbar
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.12)  # 底部预留15%空间给colorbar
        # 创建colorbar专用axes（左下角坐标，宽度，高度）
        cbar_ax = fig.add_axes([0.15, 0.08, 0.7, 0.03])  # [左，下，宽，高]
        fig.colorbar(im, 
                    cax=cbar_ax, 
                    orientation='horizontal')
        plt.show()

