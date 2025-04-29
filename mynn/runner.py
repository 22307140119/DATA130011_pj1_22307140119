import numpy as np
import os
from tqdm import tqdm
from .models import *

class RunnerM():
    """
    This is an exmaple to train, evaluate, save, load the model. However, some of the function calling may not be correct 
    due to the different implementation of those models.
    """
    def __init__(self, model, optimizer, metric, loss_fn, batch_size=32, scheduler=None, data_augmentor=None, image_shape=None):
        self.model = model          # models.py
        self.optimizer = optimizer  # optimizer.py
        self.loss_fn = loss_fn      # op.py
        self.metric = metric        # metric
        self.scheduler = scheduler  # lr_scheduler.py
        self.data_augmentor = data_augmentor    # data_augmentation.py
        self.image_shape = image_shape          # for data augmentation
        self.batch_size = batch_size

        # 训练集/验证集 的 得分/损失
        self.train_scores = []
        self.dev_scores = []
        self.train_loss = []
        self.dev_loss = []


    def train(self, train_set, dev_set, **kwargs):
        num_epochs = kwargs.get("num_epochs", 0)    # 从 kwargs 字典中获取键为 "num_epochs" 的值，获取不到则设置为 0
        log_iters = kwargs.get("log_iters", 100)    # 获取键为 "log_iters" 的值，默认值为 100 （每训练多少个 batch 输出一下）
        save_dir = kwargs.get("save_dir", "best_model") # 获取键为 "save_dir" 的值，默认值为 "best_model"

        if not os.path.exists(save_dir):    # 路径不存在则自动创建
            os.mkdir(save_dir)

        best_score = 0

        for epoch in range(num_epochs):
            # 一个 epoch
            X, y = train_set

            assert X.shape[0] == y.shape[0]

            idx = np.random.permutation(range(X.shape[0]))  # 重排序

            X = X[idx]
            y = y[idx]

            for iteration in range(int(X.shape[0] / self.batch_size) + 1):
                # 一个 batch
                train_X = X[iteration * self.batch_size : (iteration+1) * self.batch_size].copy()  # 用深拷贝，防止数据增强修改原本的训练集 X
                train_y = y[iteration * self.batch_size : (iteration+1) * self.batch_size]
                if self.data_augmentor:             # 作用动态数据增强
                    original_shape = train_X.shape
                    train_X = train_X.reshape(original_shape[0], *self.image_shape)
                    train_X = self.data_augmentor.data_augmentation(train_X)
                    train_X = train_X.reshape(original_shape)

                logits = self.model(train_X)        # 模型 model 的前向传播
                trn_loss = self.loss_fn(logits, train_y)    # 计算训练集损失 (batch)
                self.train_loss.append(trn_loss)
                
                trn_score = self.metric(logits, train_y)    # 计算训练集正确率 (batch)
                self.train_scores.append(trn_score)

                # the loss_fn layer will propagate the gradients.
                self.loss_fn.backward()     # 反向传播，得到梯度

                self.optimizer.step()       # 用上面的梯度，更新参数
                if self.scheduler is not None:
                    self.scheduler.step()   # 每一个步更新后，都看 lr 是否需要修改
                
                if isinstance(self.model, Model_MLP):               # MLP，每个 batch 参数更新后，计算开发集损失和正确率
                    dev_score, dev_loss = self.evaluate(dev_set)
                    self.dev_scores.append(dev_score)
                    self.dev_loss.append(dev_loss)

                if isinstance(self.model, Model_CNN):               # CNN，每次输出 log 前，计算开发集损失和正确率
                    if (iteration) % log_iters == 0:
                        dev_score, dev_loss = self.evaluate(dev_set)
                        self.dev_scores.append(dev_score)
                        self.dev_loss.append(dev_loss)

                if (iteration) % log_iters == 0:    # 每训练 log_iters 个 batch 就输出一下 loss
                    # 添加了输出损失项
                    L2_loss = L2_regularization(self.model)
                    print(f"epoch: {epoch}, iteration (batch): {iteration}")
                    print(f"L2 regularization term (ignore it if no weight_decay is set True): {L2_loss}")
                    print(f"[Train] loss: {trn_loss}, score: {trn_score}")
                    print(f"[Dev] loss: {dev_loss}, score: {dev_score}")

            if dev_score > best_score:      # 一个 epoch 结束，根据开发集正确率决定是否保存（覆盖）模型
                save_path = os.path.join(save_dir, 'best_model.pickle')
                self.save_model(save_path)  # 保存模型
                print(f"best accuracy performence has been updated: {best_score:.5f} --> {dev_score:.5f}")
                best_score = dev_score
        self.best_score = best_score


    def evaluate(self, data_set):   # 用于获取开发集的正确率和损失（训练集的直接在前向传播过程中得到了）
        X, y = data_set
        # logits = self.model(X)
        logits = self.model.predict(X)
        # print(f'runner.py: evaluate: logits = {logits}')
        loss = self.loss_fn(logits, y)
        score = self.metric(logits, y)
        return score, loss


    def save_model(self, save_path):
        self.model.save_model(save_path)