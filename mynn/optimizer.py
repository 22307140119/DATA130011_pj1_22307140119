from abc import abstractmethod
import numpy as np


class Optimizer:
    def __init__(self, init_lr, model) -> None:
        self.init_lr = init_lr
        self.model = model

    @abstractmethod # 基类 optimizer 不能拥有实例
    def step(self):
        pass


class SGD(Optimizer):
    def __init__(self, init_lr, model):
        super().__init__(init_lr, model)
    
    def step(self):
        for layer in self.model.layers:
            if layer.optimizable == True:   # 具有可优化参数
                for key in layer.params.keys(): # layer.params 是字典，用 .keys() 获取字典的键 'W', 'b'
                    if layer.weight_decay:
                        layer.params[key] *= (1 - self.init_lr * layer.weight_decay_lambda) # L2 正则化
                    layer.params[key] = layer.params[key] - self.init_lr * layer.grads[key] # 用传播下来的梯度


class MomentGD(Optimizer):
    def __init__(self, init_lr, model, mu=0.9): # 额外提供参数 mu
        super().__init__(init_lr, model)
        self.mu = mu    # 原动量项之前的系数？
        self.v = []     # 存放各个参数动量项的列表
    
    def step(self):
        i = 0       # 当前参数在 self.v 中的索引
        for layer in self.model.layers:
            if layer.optimizable == True:   # 具有可优化参数
                for key in layer.params.keys(): # layer.params 是字典，用 .keys() 获取字典的键 'W', 'b'

                    # L2 正则化
                    if layer.weight_decay:
                        layer.params[key] *= (1 - self.init_lr * layer.weight_decay_lambda)
                    
                    # Momentum GD
                    if len(self.v) <= i:    # 当前参数还没有历史动量记录，则初始化对应的动量项为全零
                        self.v.append(np.zeros_like(layer.params[key]))
                    self.v[i] = self.mu * self.v[i] - self.init_lr * layer.grads[key]   # 计算新的动量
                    layer.params[key] = layer.params[key] + self.v[i]                   # 更新
                    i += 1

