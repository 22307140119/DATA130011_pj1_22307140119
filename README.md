
### 1 代码架构
#### 1.1 mynn
- mynn: op.py 
  - 定义激活函数/各种层/损失函数。
  - 实现了全连接层、卷积层、池化层、展平层、ReLU 激活函数、交叉熵损失、softmax()、L2正则化项计算、
- mynn: models.py 
  - 定义模型。
  - 实现了 MLP 和 CNN 模型，注意两个模型的定义方式不同。
- mynn: optimizer.py 
  - 定义优化器。
  - 实现了 SGD 和 Moment GD。
- mynn: lr_scheduler.py 
  - 定义学习率调度策略。
  - 实现了 ExponentialLR, StepLR 和 MultiStepLR。
- mynn: data_augmentation.py 
  - 定义数据增强。
  - 支持“平移+旋转+缩放”的数据增强。
- mynn: runner.py 
  - 定义训练器。
- mynn: op_slowCNN.py 
  - 含有用多层嵌套循环实现的效率较低的卷积层和池化层。
  - op.py 中的卷积层和池化层是经过向量化优化的。

#### 1.2 其它文件
- test_train.py 训练MLP。
- test_train_CNN.py 训练CNN。
- test_model.py 测试MLP。
- test_model_CNN.py 测试CNN。
- visualization_MLP.py 可视化MLP。
- visualization_CNN.py 可视化CNN。


### 模型训练
test_train.py 用于训练 MLP ， test_train_CNN.py 用于训练 CNN。
```python
import mynn as nn
```

#### 定义单层 layer
对应 op.py 。
##### 全连接层
```python
nn.op.Linear(in_dim, out_dim, initialize_method=np.random.normal, weight_decay=False, weight_decay_lambda=1e-8)
```
##### 卷积层
```python
nn.op.conv2D(in_channels, out_channels, kernel_size, stride=1, padding=0, initialize_method='Kaiming', weight_decay=False, weight_decay_lambda=1e-8)
```
##### 池化层
```python
nn.op.MaxPooling(kernel_size=2, stride=2, padding=0)
```
##### 展平层
把 (B, C, H, W) 的输入展平为 (B, C\*H\*W)。
```python
nn.op.Flatten()
```
##### 激活函数 ReLU
```python
nn.op.ReLU()
```

#### 定义模型 model
对应 models.py 。
##### MLP 模型
无需预先定义层，直接定义模型。
```python
nn.models.Model_MLP(input_dim=None, nHidden=None, output_dim=None, act_func=None, lambda_list=None, initialization_method=np.random.normal, dropout_p=1)
```
##### CNN 模型
需要先定义各个层 layers ，定义到损失函数之前（不含损失函数）即可。
```python
nn.models.Model_CNN(layers=None, dropout_p=1)
```

#### 定义优化器 optimizer
对应 optimizer.py 。
##### SGD 优化器
```python
nn.optimizer.SGD(init_lr, model)
```
model 为先前定义的模型。
##### MomentGD 优化器
```python
nn.optimizer.MomentGD(init_lr, model, mu=0.9)
```
model 为先前定义的模型， mu 为动量项前的系数。

#### 定义学习率调度策略 scheduler
对应 lr_scheduler.py 。
##### ExponentialLR
每步的学习率乘以固定的衰减因子 gamma 。
```python
nn.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)
```
optimizer 为先前定义的优化器。
##### StepLR
每走 step_size 步，降低学习率为原本的 gamma 倍。
```python
nn.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
```
optimizer 为先前定义的优化器。
##### MultiStepLR
在预定义的多个训练阶段 milestones 中，按照固定比例 gamma 衰减学习率。
```python
nn.lr_scheduler.MultiStepLR(optimizer, milestones, gamma)
```
optimizer 为先前定义的优化器。

#### 定义损失函数 loss_fn
对应 op.py 。
##### 交叉熵损失
```python
nn.op.MultiCrossEntropyLoss(model=None, max_classes=10)
```
model 为先前定义的模型， max_classes 为分类的类别数量。

#### 定义数据增强 data_augmentor
对应 data_augmentation.py
##### 支持“平移+旋转+缩放”的数据增强
```python
nn.data_augmentation.MyDataAugmentor(max_shift_h=3, max_shift_w=3, max_angle=0.1, scale_range=(0.9, 1.1), prob=0.5, padding=0)
```
max_shift_h, max_shift_w 是平移的最大像素数量，max_angle 是旋转的最大弧度数，scale_range 是缩放区间。
prob 表示图像被增强的概率。<br>
注意如果使用静态数据增强，需要在开始训练前立刻对数据作用数据增强，例如：
```python
train_imgs = data_augmentor.data_augmentation(train_imgs)
```

#### 定义训练器 runner
对应 runner.py 。
```python
nn.runner.RunnerM(model, optimizer, metric, loss_fn, batch_size=32, scheduler=None, data_augmentor=None, image_shape=None)
```
model, optimizer, loss_fn, scheduler, data_augmentor 为先前定义的模型、优化器、损失函数、学习率策略、数据增强。<br>
data_augmentor 仅在使用动态数据增强的情况下需要定义。不使用数据增强或使用静态数据增强的情况下，不需要设置该参数，或者设置为 None 即可。<br>
metric 为评估准确率的方法，给定预测标签和真实标签，返回预测准确率。<br>
image_shape 为单张图像的 (C, H, W) 尺寸，该参数仅在 data_augmentor 非空时有用。

#### 启动训练
```python
runner.train(train_set, dev_set, num_epochs=0, log_iters=100, save_dir='best_model')
```
对 MLP ， runner.train 会在每一个 batch 后在 dev_set 上测试准确率。每过 log_iters 个 batch 在终端输出信息。
对 CNN ， runner.train 会每过 log_iters 个 batch 后再 dev_set 上测试准确率，然后在终端输出信息。


### 3 模型测试
根据模型类型为 MLP / CNN ，在 test_model.py / test_model_CNN.py 中修改 load_model() 中的模型路径后运行即可。


### 4 可视化
报告中第 8 节的可视化由 visualization_MLP.py 和 visualization_CNN.py 实现。
由于时间仓促，只实现了针对特定结构的模型在MNIST数据集上的可视化，这里不对使用方法另加说明。

