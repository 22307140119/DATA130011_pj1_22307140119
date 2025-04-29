from abc import abstractmethod
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

class Layer():
    def __init__(self) -> None:
        self.optimizable = True
    
    @abstractmethod
    def forward():
        pass

    @abstractmethod
    def backward():
        pass



class Linear(Layer):
    """
    The linear layer for a neural network. You need to implement the forward function and the backward function.
    """
    def __init__(self, in_dim, out_dim, initialize_method=np.random.normal, weight_decay=False, weight_decay_lambda=1e-8) -> None:
        super().__init__()

        # 初始化参数
        if initialize_method == 'Kaiming' or initialize_method == 'He': # Kaiming 初始化，别称 He 初始化
            self.W = np.random.randn(in_dim, out_dim) * np.sqrt(2. / in_dim)
            self.b = np.zeros(out_dim)
        else:
            self.W = initialize_method(size=(in_dim, out_dim))
            # self.b = initialize_method(size=(1, out_dim))
            self.b = initialize_method(size=(out_dim))
        
        self.grads = {'W' : None, 'b' : None}   # gradient
        self.input = None # Record the input for backward process.
        self.params = {'W' : self.W, 'b' : self.b}

        self.weight_decay = weight_decay # whether using weight decay
        self.weight_decay_lambda = weight_decay_lambda # control the intensity of weight decay
            
    
    def __call__(self, X) -> np.ndarray:
        return self.forward(X)

    def forward(self, X):
        """
        W: [in_dim, out_dim]
        input X: [batch_size, in_dim]
        out: [batch_size, out_dim]
        """
        # 先把上一轮更新后的 W 和 b 从字典中取出来
        self.W = self.params["W"]
        self.b = self.params["b"]
        self.input = X

        # 检查维数
        # print(f'op.py: Linear.forward(): X.shape = {X.shape}')
        _, Din = X.shape
        in_dim, _ = self.W.shape
        assert Din == in_dim

        return np.dot(X, self.W) + self.b
        # pass

    def backward(self, grad : np.ndarray):
        """
        input: [batch_size, out_dim] the grad passed by the next layer.
        output: [batch_size, in_dim] the grad to be passed to the previous layer.
        grad: [batch_size, out_dim]
        This function also calculates the grads for W and b. 
        """
        # weight decay 已经在 optimizer 中处理了
        # 处理 weight decay，直接把传到 优化器 中的参数 在被传入优化器之前 乘以(1-λ)
        # if self.weight_decay:
        #     self.params["W"] *= (1-self.weight_decay_lambda)
        #     self.params["b"] *= (1-self.weight_decay_lambda)

        # 检查维数
        _, Dout = grad.shape
        _, out_dim = self.W.shape
        assert Dout == out_dim

        # 梯度应该放在 self.grads 中，用和 self.params 相似的字典形式
        self.grads['W'] = np.dot(self.input.T, grad)
        self.grads['b'] = np.mean(grad, axis=0, keepdims=True)  # grad 在 batchsize 维数上取平均（损失函数中取平均）为 [1, dout]
        # d( XW + b ) / dX
        return np.dot(grad, self.W.T)
        # pass
    
    def clear_grad(self):
        self.grads = {'W' : None, 'b' : None}



class conv2D(Layer):
    """
    The 2D convolutional layer. Try to implement it on your own.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, initialize_method='Kaiming', weight_decay=False, weight_decay_lambda=1e-8) -> None:
        super().__init__()

        self.in_channels = in_channels      # 输入的通道数（卷积核的通道数）
        self.out_channels = out_channels    # 输出的通道数（卷积核的数量）
        self.kernel_size = kernel_size      # 卷积核的边长

        self.stride = stride
        self.padding = padding
        self.weight_decay = weight_decay
        self.weight_decay_lambda = weight_decay_lambda
        
        # 原本的初始化方法不太行……
        if initialize_method == 'Kaiming' or initialize_method == 'He': # Kaiming 初始化，别称 He 初始化
            self.W = np.random.randn(out_channels, in_channels, kernel_size, kernel_size) * np.sqrt(2. / (in_channels * kernel_size**2))
            self.b = np.zeros(out_channels) 
        else:
            self.W = initialize_method(size=(out_channels, in_channels, kernel_size, kernel_size))
            self.b = initialize_method(size=(out_channels))

        self.grads = {'W' : None, 'b' : None}   # gradient
        # self.input = None # Record the input for backward process.
        self.cache = None
        self.params = {'W' : self.W, 'b' : self.b}


    def __call__(self, X) -> np.ndarray:
        return self.forward(X)


    def forward(self, X):
        """
        input X: [batch_size, in_channels, H, W]
        W : [out_channels, in_channels, k, k]
        output: [batch_size, in_channels, new_H, new_W]
        no padding (? do you mean zero padding ?)
        """

        # 辅助函数 im2col
        # 将卷积核在输入中对应的每一个区域都展开成列，长度 C*kernel_size*kernel_size
        # 返回 cols: [B, (C*kernel_size*kernel_size), (H_out*W_out)]
        # 最后一维 H_out*W_out 对应“输出”的每一个元素
        # 第二维 C*kernel_size*kernel_size 对应“卷积核”的每一个元素
        # 对卷积核 W 求梯度：固定第二维(对单个卷积核元素)，看到 H_out*W_out 个元素，即为和被固定的卷积核元素相关的(做内积的)输入元素
        # 一个输入元素可能会被重复存好多遍(做一次卷积就存一遍)，空间复杂度变大了，但是后续运算的时间缩短了
        def im2col(X, kernel_size, stride, padding):
            B, C, H, W = X.shape
            X_padded = np.pad(X, [(0,0), (0,0), (padding, padding), (padding, padding)], mode='constant')
            H_out = (H + 2*padding - kernel_size) // stride + 1
            W_out = (W + 2*padding - kernel_size) // stride + 1

            # 使用滑动窗口视图 (这样就不用循环了)
            windows = sliding_window_view(X_padded, (C, kernel_size, kernel_size), axis=(1, 2, 3))
            # 处理步长, windows: [B, H_out, W_out, C, kernel_size, kernel_size]
            windows = windows[:, ::stride, ::stride, :, :, :]
            cols = windows.reshape(B, H_out*W_out, C*kernel_size*kernel_size).transpose(0, 2, 1)

            return cols
        

        # 仍然先把上一轮更新后的 W 和 b 从字典中取出来
        self.W = self.params["W"]
        self.b = self.params["b"]

        # 将卷积核在输入中对应的每一个区域都展开成列，长度 C*kernel_size*kernel_size
        # cols: [B, in_channels*kernel_size*kernel_size, H_out*W_out]
        cols = im2col(X, self.kernel_size, self.stride, self.padding)

        # W_reshaped: [out_channels, in_channels*kernel_size*kernel_size] ，后续和 cols 做矩阵乘法
        # cols: [B, in_channels*kernel_size*kernel_size, H_out*W_out]
        # W_reshaped @ cols 只要 W_reshaped 的第二个维数和 cols 的倒数第二个维数相同就可以
        # output: [B, out_channels, H_out*W_out]
        W_reshaped = self.W.reshape(self.out_channels, self.in_channels*self.kernel_size*self.kernel_size)
        output = W_reshaped @ cols + self.b.reshape(self.out_channels, 1)
        # print(cols.shape, W_reshaped.shape, output.shape)

        # output 转换为输出形状 [B, out_channels, H_out, W_out]
        B, _, H, W = X.shape
        H_out = (W + 2*self.padding - self.kernel_size) // self.stride + 1
        W_out = (W + 2*self.padding - self.kernel_size) // self.stride + 1
        output = output.reshape(B, self.W.shape[0], H_out, W_out)

        # 缓存一下数据，反向传播里面用
        self.cache = (X, cols)
        return output
    

    def backward(self, grads):
        """
        grads : [batch_size, out_channels, new_H, new_W]
        input X: [batch_size, in_channels, H, W]
        W : [out_channels, in_channels, k, k]
        """

        # 辅助函数 col2im
        # dcols 存储一些梯度值，大小 [B, C*kernel_size*kernel_size, H_out*W_out]
        # 求和消除 dcols 中的冗余元素，返回梯度 dX ，大小和输入的 X 相同
        # 先加 padding (求和用)，最后去除 padding
        def col2im(dcols, X_shape, kernel_size, stride, padding):
            B, C, H, W = X_shape
            H_padded = H + 2 * padding
            W_padded = W + 2 * padding
            dX_padded = np.zeros((B, C, H_padded, W_padded))
            
            H_out = (H_padded - kernel_size) // stride + 1
            W_out = (W_padded - kernel_size) // stride + 1
            
            # dcols: [B, (C*kernel_size*kernel_size), (H_out*W_out)]
            for h in range(H_out):
                for w in range(W_out):
                    h_start = h * stride
                    h_end = h_start + kernel_size
                    w_start = w * stride
                    w_end = w_start + kernel_size

                    # 固定 dcols 的最后一个维度，提取出 [B, (C*kernel_size*kernel_size)]，转换成 [B, C, kernel_size, kernel_size]
                    # patch 即参与输出元素 [h,w] 计算的输入 X 的部分的梯度值
                    patch = dcols[:, :, h*W_out+w].reshape(B, C, kernel_size, kernel_size)
                    dX_padded[:, :, h_start:h_end, w_start:w_end] += patch
            
            # 去除 padding
            if padding > 0:
                return dX_padded[:, :, padding:-padding, padding:-padding]
            else:
                return dX_padded


        # cols: [B, in_channels*kernel_size*kernel_size, H_out*W_out]
        X, cols = self.cache
        B = X.shape[0]

        # 计算 dW 和 db
        # 'bom,bim->oi' 相当于对 o,i 循环: dW[o,i] = Σ_b Σ_m (grads_flat[b,o,m] * cols[b,i,m])
        # 相当于固定第二维(对卷积核求梯度)，用 col 的最后一维(卷积核的每一个元素对应的输入元素)和梯度 grads_flat 做内积，再对每一个 batch 求和
        # grads_flat: [B, out_channels, H_out*W_out]
        # np.einsum() 维数: [out_channels, in_channels*kernel_size*kernel_size]
        grads_flat = grads.reshape(B, self.out_channels, -1)
        dW = np.einsum('bom,bim->oi', grads_flat, cols).reshape(self.W.shape)
        # db: 对 grads 的第 0, 2, 3 维数求和，维数: [out_channels]
        db = np.sum(grads, axis=(0,2,3))
        self.grads['W'] = dW
        self.grads['b'] = db

        # 计算 dX
        # W_reshaped: [out_channels, in_channels*kernel_size*kernel_size]
        # grads_flat: [B, out_channels, H_out*W_out]
        # 梯度 dcols: [B, in_channels*kernel_size*kernel_size, H_out*W_out]
        # dcols 中的每一个元素均存储 某一个卷积核元素和某一个梯度值的乘积
        W_reshaped = self.W.reshape(self.W.shape[0], -1)
        dcols = W_reshaped.T @ grads_flat
        # print(dcols.shape, self.W.shape, W_reshaped.shape, grads_flat.shape)
        # 将梯度 dcols 转换回输入形状
        dX = col2im(dcols, X.shape, self.kernel_size, self.stride, self.padding)
        return dX


    def clear_grad(self):
        self.grads = {'W' : None, 'b' : None}



class MaxPooling(Layer):
    def __init__(self, kernel_size=2, stride=2, padding=0):
        super().__init__()
        self.optimizable = False
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        
        self.max_h = None    # 最大值所在行坐标（填充后）
        self.max_w = None    # 最大值所在列坐标（填充后）
        self.H_pad = None    # 填充后的高度
        self.W_pad = None    # 填充后的宽度


    def __call__(self, X):
        return self.forward(X)


    def forward(self, X):
        '''
        input X: [batch_size, channels, H, W]
        '''
        # padding
        X_padded = np.pad(X, 
                         [(0,0), (0,0), 
                         (self.padding, self.padding), 
                         (self.padding, self.padding)],
                         mode='constant', constant_values=-np.inf)
        
        B, C, H_pad, W_pad = X_padded.shape
        self.H_pad, self.W_pad = H_pad, W_pad
        
        # 计算输出尺寸
        H_out = (H_pad - self.kernel_size) // self.stride + 1
        W_out = (W_pad - self.kernel_size) // self.stride + 1
        
        # 提取滑动窗口 [B, C, H_out, W_out, K, K]
        windows = sliding_window_view(X_padded, (self.kernel_size, self.kernel_size), axis=(2,3))
        windows = windows[:, :, ::self.stride, ::self.stride, :, :]
        
        # 展平窗口最后一维 [B,C,H_out,W_out,K*K]
        flat_windows = windows.reshape(*windows.shape[:-2], self.kernel_size**2)
        
        # 计算最大值索引
        max_indices = flat_windows.argmax(axis=-1)   # [B, C, H_out, W_out]
        rows = max_indices // self.kernel_size       # 窗口内的行偏移
        cols = max_indices % self.kernel_size        # 窗口内的列偏移
        
        # 计算窗口起始坐标
        h_start = np.arange(H_out).reshape(1, 1, H_out, 1) * self.stride  # [1, 1, H_out, 1]
        w_start = np.arange(W_out).reshape(1, 1, 1, W_out) * self.stride  # [1, 1, 1, W_out]
        
        # 计算实际坐标
        self.max_h = h_start + rows  # [B, C, H_out, W_out] ，指示输出中每一个元素对应的输入元素的横坐标
        self.max_w = w_start + cols  # [B, C, H_out, W_out] ，指示输出中每一个元素对应的输入元素的纵坐标
        
        return flat_windows.max(axis=-1)  # 最大值结果


    def backward(self, grads):
        '''
        grads: [batch_size, channels, H_out, W_out]
        '''
        B, C, H_out, W_out = grads.shape
        
        # 初始化梯度容器（填充后尺寸）
        dX_padded = np.zeros((B, C, self.H_pad, self.W_pad), dtype=grads.dtype)
        
        # 生成 batch (B) 和 channel (C) 的索引模板，用于后面的 np.add.at()
        # [B, C, H_out, W_out] ，第一维是表示 batch num 的连续整数
        batch_idx = np.broadcast_to(np.arange(B)[:, None, None, None], (B, C, H_out, W_out))
        # [B, C, H_out, W_out] ，第二维是表示 channel num 的连续整数
        channel_idx = np.broadcast_to(np.arange(C)[None, :, None, None], (B, C, H_out, W_out))
        
        # 使用 np.add.at 进行梯度累加
        np.add.at(dX_padded, (batch_idx, channel_idx, self.max_h, self.max_w), grads)
        
        # 移除 padding
        if self.padding > 0:
            dX = dX_padded[:, :, self.padding:-self.padding, self.padding:-self.padding]
        else:
            dX = dX_padded
        
        return dX



class ReLU(Layer):
    """
    An activation layer.
    """
    def __init__(self) -> None:
        super().__init__()
        self.input = None

        self.optimizable =False

    def __call__(self, X):
        return self.forward(X)

    def forward(self, X):
        self.input = X
        output = np.where(X<0, 0, X)
        return output
    
    def backward(self, grads):
        assert self.input.shape == grads.shape
        output = np.where(self.input < 0, 0, grads)
        return output



class Flatten(Layer):
    def __init__(self):
        super().__init__()
        self.optimizable = False
        self.input_shape = None
    
    def __call__(self, X):
        return self.forward(X)

    def forward(self, X):
        self.input_shape = X.shape  # 保存原始形状
        bs, c, h, w = X.shape
        return X.reshape(bs, c*h*w)
    
    def backward(self, grads):
        return grads.reshape(self.input_shape)



class MultiCrossEntropyLoss(Layer):
    """
    A multi-cross-entropy loss layer, with Softmax layer in it, which could be cancelled by method cancel_softmax
    """
    def __init__(self, model=None, max_classes=10) -> None:
        super().__init__()
        self.optimizable = False
        self.input = None   # for backward process
        self.probs = None   # for backward process
        self.labels = None  # for backward process

        self.model = model
        self.max_classes = max_classes
        self.has_softmax = True
        self.grads = None
        # pass

    def __call__(self, predicts, labels):
        return self.forward(predicts, labels)
    
    def forward(self, predicts, labels):
        """
        predicts(input): [batch_size, D]
        labels : [batch_size, ]
        self.probs: [batch_size, D]
        correct_probs: [batch_size, ] 
        This function generates the loss.
        """
        # / ---- your codes here ----/
        self.input = predicts
        self.labels = labels
        if self.has_softmax:
            self.probs = softmax(self.input)
            # print(f'op.py: MultiCrossEntropyLoss.forward(): probs = {self.probs}')
        else:
            self.probs = self.input
        
        # CrossEntropy 用平均损失，为了防止 log(0) 加上小量 1e-15
        batch_size = self.probs.shape[0]
        correct_probs = self.probs[np.arange(batch_size), labels]
        loss = -np.mean(np.log(correct_probs + 1e-15))

        # L2 正则化项，还是不加为好，加上以后损失变得很大了
        # 但是可以用 L2 正则化项看模型的参数正常不正常（参数可能会变得太接近0）
        # loss += L2_regularization(self.model)

        return loss
        # pass
    
    def backward(self):
        # first compute the grads from the loss to the input
        # / ---- your codes here ----/
        one_hot_labels = np.eye(self.max_classes)[self.labels]  # 从单位矩阵中选取 [self.labels] 指定的行，拼起来
        batch_size = self.input.shape[0]

        if self.has_softmax:
            # softmax + cross entropy 简化后
            self.grads = (self.probs - one_hot_labels) / batch_size
        else:
            # cross entropy
            self.grads = - (one_hot_labels / (self.probs + 1e-15)) / batch_size

        # Then send the grads to model for back propagation
        self.model.backward(self.grads)

    def cancel_soft_max(self):
        self.has_softmax = False
        return self



class L2Regularization(Layer):
    """
    L2 Reg can act as weight decay that can be implemented in class Linear.
    """
    pass



def L2_regularization(model):   # 返回 L2 损失项
    # 参考了 optimizer.py
    res = 0
    for layer in model.layers:  # 遍历所有层
        if layer.optimizable == True:           # 如果层是可优化的
                for key in layer.params.keys(): # 获取字典的键
                    if layer.weight_decay:      # 如果层需要进行 weight_decay ，计算二范数 * λ
                        res += np.sum(layer.params[key]**2) * layer.weight_decay_lambda
    return res


def softmax(X):
    x_max = np.max(X, axis=1, keepdims=True)
    x_exp = np.exp(X - x_max)
    partition = np.sum(x_exp, axis=1, keepdims=True)
    return x_exp / partition



# 调试用
if __name__ == '__main__':

    # 卷积层
    conv = conv2D(in_channels=3, out_channels=5, kernel_size=4, padding=2)
    X = np.random.randn(2, 3, 32, 32)  # 输入数据

    # 前向传播
    output = conv(X)
    # print(output)
    print(output.shape)

    # 反向传播
    grads = np.ones_like(output)    # 生成全一的上游梯度，大小和 output 一样
    dX = conv.backward(grads)
    # print(grads, dX)
    print(grads.shape, dX.shape)


    # 池化层
    X = output

    pool = MaxPooling(kernel_size=2, stride=2, padding=1)
    output = pool.forward(X)
    # print(output)
    print(output.shape)

    grads = np.ones_like(output)
    dX = pool.backward(grads)
    # print(grads, dX)
    print(grads.shape, dX.shape)

