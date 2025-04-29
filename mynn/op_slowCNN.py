from abc import abstractmethod
import numpy as np

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
        self.W = initialize_method(size=(in_dim, out_dim))
        self.b = initialize_method(size=(1, out_dim))
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
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, initialize_method=np.random.normal, weight_decay=False, weight_decay_lambda=1e-8) -> None:
        super().__init__()

        self.in_channels = in_channels      # 输入的通道数（卷积核的通道数）
        self.out_channels = out_channels    # 输出的通道数（卷积核的数量）
        self.kernel_size = kernel_size      # 卷积核的边长

        self.stride = stride
        self.padding = padding
        self.weight_decay = weight_decay
        self.weight_decay_lambda = weight_decay_lambda

        # self.W = initialize_method(size=(out_channels, in_channels, kernel_size, kernel_size))  # 初始化 W
        # self.b = initialize_method(size=(out_channels))   # 初始化 b，前向传播时 b 被广播成 new_H * new_W 的大小，加在每一个卷积核内积的结果上
        # 原本的初始化方法不行，只能用这个
        self.W = np.random.randn(out_channels, in_channels, kernel_size, kernel_size) * np.sqrt(2. / (in_channels * kernel_size**2))
        self.b = np.zeros(out_channels) 
        self.grads = {'W' : None, 'b' : None}   # gradient
        self.input = None # Record the input for backward process.
        self.params = {'W' : self.W, 'b' : self.b}
    

    def __call__(self, X) -> np.ndarray:
        return self.forward(X)
    

    # 卷积层的前向传播
    def forward(self, X):
        """
        input X: [batch_size, in_channels, H, W]
        W : [1, out_channels, in_channels, k, k]
        output: [batch_size, in_channels, new_H, new_W]
        no padding (? do you mean zero padding ?)
        """
        # 仍然先把上一轮更新后的 W 和 b 从字典中取出来
        self.W = self.params["W"]
        self.b = self.params["b"]

        batch_size, Cin, H, W = X.shape
        assert Cin == self.in_channels      # 检查输入的通道数
        k = self.kernel_size
        self.input = X  # 用于 backward()

        # 输出的 H 和 W
        new_H = (H + 2*self.padding - k) // self.stride + 1
        new_W = (W + 2*self.padding - k) // self.stride + 1
        # 初始化输出，输出尺寸 (batch_size, out_channels, new_H, new_W)
        output = np.zeros((batch_size, self.out_channels, new_H, new_W))

        # padding
        X_padded = np.pad(X, ((0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding)), mode='constant')


        # 卷积，但是好像只能用嵌套循环，逐元素计算
        # print(f'op.py: conv2D.forward(): X.shape: {X.shape}')
        for batch in range(batch_size):
            # print(f'message from conv2D forward(), {batch}')
            for keridx in range(self.out_channels):     # 遍历每一个卷积核
                for h_new in range(new_H):      # 输出的行
                    for w_new in range(new_W):  # 输出的列
                        # 提取 X_padded 中，要和卷积核做内积的部分
                        h0 = h_new * self.stride
                        w0 = w_new * self.stride
                        window = X_padded[batch, :, h0:h0+k, w0:w0+k]

                        # 计算内积，加上偏置，填到输出的对应位置
                        output[batch, keridx, h_new, w_new] = np.sum(window * self.W[keridx]) + self.b[keridx]

        return output


    # 卷积层的反向传播
    def backward(self, grads):
        """
        grads : [batch_size, out_channels, new_H, new_W]
        """
        batch_size, Cout, Hout, Wout = grads.shape
        assert Cout == self.out_channels
        k = self.kernel_size

        # 检查一下长宽是否匹配（主要检查上游梯度 grads 的维数计算是否正确）
        batch_size, Cin, H, W = self.input.shape
        new_H = (H + 2*self.padding - k) // self.stride + 1
        new_W = (W + 2*self.padding - k) // self.stride + 1
        assert new_H == Hout
        assert new_W == Wout

        # padding
        X_padded = np.pad(self.input, ((0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding)), mode='constant')

        # 直接求出 b 的梯度，对 grads 中除了 out_channels 的所有维度求和，得到维数为 out_channels 的向量，就是 b 的梯度
        db = np.sum(grads, axis=(0, 2, 3))

        # 初始化 W 和 X_padded 的梯度为全 0 ，后面慢慢累加……
        dW = np.zeros_like(self.W)
        dX_padded = np.zeros_like(X_padded)

        # 计算损失 W 和 X_padded 的梯度，还是从 output 的逐元素视角看，可以直接对应到一个上游梯度
        for batch in range(batch_size):
            for keridx in range(self.out_channels):
                for h_new in range(new_H):
                    for w_new in range(new_W):
                        # 提取和输出中 (batch, c_out, h_new, w_new) 元素 相关联的 X 的窗口
                        h_start = h_new * self.stride
                        w_start = w_new * self.stride
                        window = X_padded[batch, :, h_start:h_start+k, w_start:w_start+k]
                        grad_val = grads[batch, keridx, h_new, w_new]   # 对应的上游梯度
                        # 更新卷积核的梯度 (更新一整个卷积核，更新量为 窗口 * 上游梯度)
                        dW[keridx] += window * grad_val
                        
                        # 更新输入梯度 (更新大小等同于卷积核的一片区域，更新量为 卷积核 * 上游梯度)
                        dX_padded[batch, :, h_start:h_start+k, w_start:w_start+k] += self.W[keridx] * grad_val
        
        # 去除 padding 部分后，损失对 X 的梯度
        if self.padding > 0:
            dX = dX_padded[:, :, self.padding:-self.padding, self.padding:-self.padding]
        else:
            dX = dX_padded

        self.grads['W'] = dW
        self.grads['b'] = db

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
        self.input = None   # 用于反向传播
        self.mask = None    # 掩码，用于反向传播
    

    def __call__(self, X):
        return self.forward(X)
    

    def forward(self, X):
        '''
        input X: [batch_size, channels, H, W]
        '''
        batch_size, channels, H, W = X.shape
        self.input = X
        
        # 输出维数
        k = self.kernel_size
        new_H = (H + 2*self.padding - k) // self.stride + 1
        new_W = (W + 2*self.padding - k) // self.stride + 1
        # 初始化输出
        output = np.zeros((batch_size, channels, new_H, new_W))

        # padding
        X_padded = np.pad(X, ((0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding)), 'constant')
        self.mask = np.zeros_like(X_padded) # 掩码，反向传播用

        # 仍然是嵌套循环，逐个填充输出元素
        for batch in range(batch_size):
            for channel in range(channels):
                for h in range(new_H):
                    for w in range(new_W):
                        # 要取最大值的窗口位置
                        h_start = h * self.stride
                        w_start = w * self.stride
                        window = X_padded[batch, channel, h_start:h_start+k, w_start:w_start+k]
                        
                        # 取最大值，把最大值在窗口中的位置记录到 max_idx 中
                        output[batch, channel, h, w] = np.max(window)    # 最大值
                        max_idx = np.unravel_index(np.argmax(window), window.shape) # 最大值在窗口中的位置
                        
                        # 标记原始输入中最大值的位置
                        self.mask[batch, channel, h_start + max_idx[0], w_start + max_idx[1]] = 1
        
        return output


    def backward(self, grads):
        '''
        grads: [batch_size, channels, new_H, new_W]
        '''
        # 获取 batch_size, channels, new_H, new_W ，以及一些维数检查
        batch_size, channels, Hout, Wout = grads.shape
        X = self.input
        bs, c, H, W = X.shape
        assert batch_size == bs
        assert channels == c
        
        k = self.kernel_size
        new_H = (H + 2*self.padding - k) // self.stride + 1
        new_W = (W + 2*self.padding - k) // self.stride + 1
        assert new_H == Hout
        assert new_W == Wout

        # X_padded 的梯度，初始化为 0 ，大小和 self.mask 一致
        dX_padded = np.zeros_like(self.mask)
        
        # 仍然是从输出的逐元素视角
        for batch in range(batch_size):
            for channel in range(channels):
                for h in range(new_H):
                    for w in range(new_W):
                        # 找到当前的输出元素 在输入中 对应的窗口位置
                        h_start = h * self.stride
                        w_start = w * self.stride
                        window_mask = self.mask[batch, channel, h_start:h_start+k, w_start:w_start+k]  # 当前窗口的掩码
                        # 窗口处对应的梯度：掩码为 0 处即为 0 ，掩码为 1 处即为上游梯度 grads 中的对应值
                        dX_padded[batch, channel, h_start:h_start+k, w_start:w_start+k] += window_mask * grads[batch, channel, h, w]

        # 去除 padding 部分后，损失对 X 的梯度
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
    def __init__(self, model = None, max_classes = 10) -> None:
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

        # L2 正则化项
        loss += L2_regularization(self.model)

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


def L2_regularization(model):
    return 0


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

