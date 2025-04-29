import numpy as np
import matplotlib.pyplot as plt
import time


class MyDataAugmentor():

    # max_shift_h, max_shift_w 是平移的最大像素数量
    # max_angle 是旋转的最大弧度数
    # scale_range 是缩放区间
    # prob 表示图像被增强的概率
    def __init__(self, max_shift_h=3, max_shift_w=3, max_angle=0.1, scale_range=(0.9, 1.1), prob=0.5, padding=0):
        self.max_shift_h = max_shift_h
        self.max_shift_w = max_shift_w
        self.max_angle = max_angle
        self.scale_range = scale_range

        self.padding = padding
        self.prob = prob
    

    # 缩放 --> 旋转 --> 平移
    def data_augmentation(self, imgs):

        if self.prob == 0:
            return imgs

        # 设置参数
        B, C, H, W = imgs.shape
        theta = (np.random.rand(B) * 2 - 1) * self.max_angle
        scale = np.random.rand(B) * (self.scale_range[1]-self.scale_range[0]) + (self.scale_range[0]+self.scale_range[1])/2
        shift_h = (np.random.rand(B) * 2 - 1) * self.max_shift_h
        shift_w = (np.random.rand(B) * 2 - 1) * self.max_shift_w

        # 作用变换
        mask = np.random.rand(B) < self.prob     # mask 数组，表示是否需要增强
        aug_imgs = scale2D(imgs, scale, self.padding)
        aug_imgs = rotate2D(aug_imgs, theta, self.padding)
        aug_imgs = shift2D(aug_imgs, shift_h, shift_w, self.padding)

        # 组合原数组和增强后的数组
        new_imgs = imgs.copy()  # 深拷贝，不会覆盖原本的 imgs 参数
        new_imgs[mask] = aug_imgs[mask]
        return new_imgs




# 双线性插值，返回和 imgs 大小相同的数组
def BiLinearInterpolation(imgs, grid_h, grid_w, padding=0):
    # imgs 图像，以 B*C*H*W 的形式给出
    # grid_h, grid_w B*C*H'*W' 数组，指定返回的数组中的每一个元素在原图像 imgs 中的位置，H',W' 不一定要和 H,W 相同
    # 一般来说给定 B 的情况下，不同 C 对应的 H*W 应该是一样的
    B, C, H, W = imgs.shape
    
    # 提取坐标的整数部分 h,w 和偏移量 dh,dw
    h = np.floor(grid_h).astype(int)
    w = np.floor(grid_w).astype(int)
    dh = grid_h - h
    dw = grid_w - w

    # 把 h,w 中不超界的元素在 h,w 中对应的下标存起来，要保证 +1 后仍然不超界
    # 最后会把结果中除了 valid 以外的元素都变成 padding
    valid = (h >= 0) & (h <= H-2) & (w >= 0) & (w <= W-2)

    # 把 h,w 中超界的下标截断成合法的 (必须在 [0, W-2] 范围内)
    h = np.clip(h, 0, H-2)
    w = np.clip(w, 0, W-2)

    # 广播索引
    b = np.arange(B)[:, None, None, None]
    c = np.arange(C)[None, :, None, None]

    # 双线性插值
    res = imgs[b, c, h, w] * (1-dh)*(1-dw)      # [h,w] * (1-dh)*(1-dw)
    res += imgs[b, c, h, w+1] * (1-dh)*dw       # [h,w+1] * (1-dh)*dw
    res += imgs[b, c, h+1, w] * dh*(1-dw)       # [h+1,w] * dh*(1-dw)
    res += imgs[b, c, h+1, w+1] * dh*dw         # [h+1,w+1] * dh*dw

    # 不合法的地方填充 padding
    # 如果 valid[i] 为 True，则结果数组中对应位置的值取自 res[i]
    # 如果 valid[i] 为 False，则结果数组中对应位置的值取自 padding[i]
    res = np.where(valid, res, padding)

    return res



# 图像的批量平移
# shift_h, shift_w 尺寸为 B, 可以取正/负实数，绝对值不宜太大（相对原图大小而言）
def shift2D(imgs, shift_h, shift_w, padding=0):
    # imgs 图像，以 B*C*H*W 的形式给出
    # shift_h, shift_w 数组，尺寸为 B ，指定每一张图片在 H,W 方向的偏移量
    # padding 指定平移后向空缺区域内填充的数据
    shift_h = np.array(shift_h)
    shift_w = np.array(shift_w)
    B, C, H, W = imgs.shape

    # 初始化，大小 B*C*H*W
    grid_h = np.zeros(imgs.shape)
    grid_w = np.zeros(imgs.shape)
    grid_h += np.arange(H).reshape(1, 1, H, 1)
    grid_w += np.arange(W).reshape(1, 1, 1, W)
    
    # grid_h, grid_w 是新图片中每一个像素对应原图片的位置，所以应该减掉
    grid_h -= shift_h.reshape(B, 1, 1, 1)
    grid_w -= shift_w.reshape(B, 1, 1, 1)

    res = BiLinearInterpolation(imgs, grid_h, grid_w, padding)
    # print('Images shifted.')

    return res


# 图像的批量旋转
# theta 尺寸为 B, 可以取正/负，绝对值不应该超过 1.57 (pi/2)
def rotate2D(imgs, theta, padding=0):
    # imgs 图像，以 B*C*H*W 的形式给出
    # theta 数组，尺寸为 B ，指定每一张图片以图像中心为原点，"顺时针"方向的旋转弧度
    # padding 指定旋转后向空缺区域内填充的数据
    theta = np.array(theta)
    B, C, H, W = imgs.shape
    
    # 初始化，大小 B*C*H*W，注意中心
    grid_h = np.zeros(imgs.shape)
    grid_w = np.zeros(imgs.shape)
    grid_h += np.arange(H).reshape(1, 1, H, 1) - (H-1)/2
    grid_w += np.arange(W).reshape(1, 1, 1, W) - (W-1)/2

    # (返回的新图像) 逆时针旋转 theta ，注意中心
    cos = np.cos(theta).reshape(B, 1, 1, 1)
    sin = np.sin(theta).reshape(B, 1, 1, 1)
    grid1_h = grid_h * cos - grid_w * sin + (H-1)/2
    grid1_w = grid_h * sin + grid_w * cos + (W-1)/2

    res = BiLinearInterpolation(imgs, grid1_h, grid1_w, padding)
    # print('Images rotated.')

    return res


# 图像的批量缩放
# scale 尺寸为 B, 只能取正值，应该在 1 附近
def scale2D(imgs, scale, padding=0):
    # imgs 图像，以 B*C*H*W 的形式给出
    # scale 数组，尺寸为 B ，指定每一张图片以图像中心为原点，"放大"的程度
    # padding 指定旋转后向空缺区域内填充的数据
    scale = np.array(scale)
    B, C, H, W = imgs.shape

    # 初始化，大小 B*C*H*W，注意中心
    grid_h = np.zeros(imgs.shape)
    grid_w = np.zeros(imgs.shape)
    grid_h += np.arange(H).reshape(1, 1, H, 1) - (H-1)/2
    grid_w += np.arange(W).reshape(1, 1, 1, W) - (W-1)/2

    # (返回的新图像) 缩小 scale 倍 ，注意中心
    scale = scale.reshape(B, 1, 1, 1)
    grid_h = grid_h / scale + (H-1)/2
    grid_w = grid_w / scale + (W-1)/2

    res = BiLinearInterpolation(imgs, grid_h, grid_w, padding)
    # print('Images scaled.')

    return res



# 调试用
if __name__ == '__main__':
    from struct import unpack
    import gzip
    import pickle
    np.random.seed(1)


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

    N_train = train_imgs.shape[0]
    train_imgs = train_imgs.reshape(N_train, 1, 28, 28)
    # print(train_imgs.shape)



    # original_imgs = train_imgs[0:5]
    # shift_imgs = shift2D(original_imgs, [-2.2, -2.5, 1.4, 2.2, 4.1], [-2.8, 2.2, -3.1, 1.8, 2.5], 0)
    # rotate_imgs = rotate2D(original_imgs, [-.7, -.5, .5, .4, .6], 0)
    # scale_imgs = scale2D(original_imgs, [.5, .8, .7, 1.2, 1.3], 0)
    # imgs = [original_imgs, shift_imgs, rotate_imgs, scale_imgs]



    original_imgs = train_imgs[0:5]
    # new_imgs = my_augmentation(original_imgs, max_shift_h=3, max_shift_w=3, max_angle=0.1, scale_range=(0.9, 1.1), prob=0.4)
    my_augmentor = MyDataAugmentor(max_shift_h=3, max_shift_w=3, max_angle=0.1, scale_range=(0.9, 1.1), prob=0.4, padding=0)
    new_imgs = my_augmentor.data_augmentation(original_imgs)
    imgs = [train_imgs[0:5], new_imgs]



    plt.figure(figsize=(8, 7))
    for i in range(10):
        plt.subplot(4, 5, i+1)
        plt.imshow(imgs[int(i/5)][i%5].reshape(28, 28))
        plt.axis('off')  # 关闭坐标轴

    plt.tight_layout()  # 调整子图间距
    plt.show()  # 显示图像

    # print(BiLinearInterpolation(imgs=imgs, grid_h=np.random.randn(5,1,1,1), grid_w=np.random.randn(5,1,1,1)))

    # 对全部的图像作用一下数据增强，看看时长
    # start = time.time()
    # trans_imgs = shift2D(train_imgs, np.random.randn(len(train_imgs)), np.random.randn(len(train_imgs)))
    # trans_imgs = rotate2D(train_imgs, np.random.randn(len(train_imgs)))
    # trans_imgs = rotate2D(train_imgs, np.random.randn(len(train_imgs))+1)
    # end = time.time()
    # print(end-start)    # 单个操作大概 8 秒

    # 先缩小再平移 优于（保留原本图像） 先平移再缩小
    # 先平移再放大 优于（保留原本图像） 先放大再平移
