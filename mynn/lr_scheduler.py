from abc import abstractmethod
import numpy as np

class scheduler():
    def __init__(self, optimizer) -> None:
        self.optimizer = optimizer
        self.step_count = 0
    
    @abstractmethod
    def step():
        pass


class StepLR(scheduler):
    def __init__(self, optimizer, step_size=30, gamma=0.1) -> None:
        super().__init__(optimizer)
        self.step_size = step_size  # 每多少步降低学习率
        self.gamma = gamma

    def step(self) -> None:
        self.step_count += 1        # 一个 batch 算一步
        if self.step_count >= self.step_size:
            self.optimizer.init_lr *= self.gamma    # 每走 step_size 步，将学习率变成原本的 gamma 倍，直接修改 optimizer 中的 init_lr
            self.step_count = 0
            print("lr adjusted.")   # 增加了调试语句


class MultiStepLR(scheduler):
    # 在预定义的多个训练阶段 milestones 中，按照固定比例 gamma 衰减学习率
    def __init__(self, optimizer, milestones, gamma):
        super().__init__(optimizer)
        self.milestones = milestones
        self.gamma = gamma
    
    def step(self) -> None:
        self.step_count += 1        # 一个 batch 算一步
        if self.step_count in self.milestones:
            self.optimizer.init_lr *= self.gamma    # 每次步数达到 milestones 中的步数，就将学习率变成原本的 gamma 倍，直接修改 optimizer 中的 init_lr
            # 无需重置 step_count
            print("lr adjusted.")   # 增加了调试语句


class ExponentialLR(scheduler):
    # 每步的学习率会乘以固定的衰减因子 gamma
    def __init__(self, optimizer, gamma=0.99):
        super().__init__(optimizer)
        self.gamma = gamma
    
    def step(self) -> None:
        self.step_count += 1
        self.optimizer.init_lr *= self.gamma    # 每一步都将学习率变成原本的 gamma 倍，直接修改 optimizer 中的 init_lr
        # print("lr adjusted.")   # 增加了调试语句