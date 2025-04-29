# codes to make visualization of your weights.
import mynn as nn
import random
import matplotlib.pyplot as plt

model = nn.models.Model_MLP()
model.load_model(r'.\best_models\best_model_9834.pickle')

w0 = model.layers[0].params['W']
w1 = model.layers[2].params['W']

# 直接可视化第一层全连接层
plt.figure(figsize=(15, 6))
plt.imshow(
    w0.T,
    cmap='bwr',      # 颜色映射
    interpolation='nearest',  # 无插值
    vmin=-0.4,
    vmax=0.4
)
plt.colorbar(label='Value')
plt.show()

# 第一层全连接层的一些行变成正方形
random.seed(2)
idxlist = [random.randint(0, 600) for _ in range(10)]
print(idxlist)
fig, axes = plt.subplots(1, len(idxlist), figsize=(15, 2))
for i in range(len(idxlist)):
    f = w0[:,idxlist[i]].reshape(28, 28)
    im = axes[i].imshow(
        f,
        # cmap='bwr',      # 颜色映射
        interpolation='nearest',  # 无插值
        vmin=-0.4,
        vmax=0.4
    )
    axes[i].axis('off')
fig.colorbar(im, ax=axes.ravel().tolist(), orientation='horizontal', pad=0.2, aspect=40)
plt.show()

# 直接可视化第二层全连接层
plt.figure(figsize=(15, 2))
plt.imshow(
    w1.T,
    cmap='bwr',      # 颜色映射
    interpolation='nearest',  # 无插值
    vmin=-0.8,
    vmax=0.8
)
plt.colorbar(label='Value', orientation='horizontal', pad=0.3, aspect=40)
plt.show()
