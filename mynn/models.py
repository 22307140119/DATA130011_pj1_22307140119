from .op import *
import pickle

class Model_MLP(Layer):
    """
    A model with linear layers. We provied you with this example about a structure of a model.
    """
    # def __init__(self, size_list=None, act_func=None, lambda_list=None):
    #     self.size_list = size_list

    def __init__(self, input_dim=None, nHidden=None, output_dim=None, act_func=None, lambda_list=None, initialization_method=np.random.normal, dropout_p=1):

        self.p = dropout_p
        if nHidden is not None and act_func is not None:

            # 用 输入维数 隐藏层维数 输出维数 重构原本的 size_list
            size_list = nHidden
            size_list.insert(0, input_dim)
            size_list.append(output_dim)
            self.size_list = size_list
            self.act_func = act_func

            self.layers = []
            for i in range(len(size_list) - 1): # size_list 中 存放每层神经元的维度
                # 线性层（全连接层）
                layer = Linear(in_dim=size_list[i], out_dim=size_list[i + 1], initialize_method=initialization_method)
                if lambda_list is not None: # 当前全连接层的 weight_decay
                    layer.weight_decay = True
                    layer.weight_decay_lambda = lambda_list[i]
                # 激活函数
                if act_func == 'Logistic':
                    raise NotImplementedError
                elif act_func == 'ReLU':
                    layer_f = ReLU()
                # 添加全连接层
                self.layers.append(layer)
                # 添加激活函数
                if i < len(size_list) - 2:  # 保证最后一层不添加激活函数
                    self.layers.append(layer_f)


    def __call__(self, X):  # 可以直接通过 实例名称(X) 调用 forward(X)
        return self.forward(X)


    def forward(self, X):
        assert self.size_list is not None and self.act_func is not None, 'Model has not initialized yet. Use model.load_model to load a model or create a new model with size_list and act_func offered.'
        outputs = X
        for layer in self.layers:
            outputs = layer(outputs)
            if isinstance(layer, ReLU): # 如果添加新的激活函数，需要在这里加上一点代码
                mask = (np.random.rand(*outputs.shape) < self.p) / self.p
                outputs *= mask
        return outputs


    def backward(self, loss_grad):
        grads = loss_grad
        for layer in reversed(self.layers):
            grads = layer.backward(grads)
        return grads


    def predict(self, X):   # 用于训练之外场合的前向传播
        assert self.size_list is not None and self.act_func is not None, 'Model has not initialized yet. Use model.load_model to load a model or create a new model with size_list and act_func offered.'
        outputs = X
        for layer in self.layers:
            outputs = layer(outputs)
        return outputs


    def load_model(self, param_list):
        with open(param_list, 'rb') as f:
            param_list = pickle.load(f)
        # 前两个元素存储 size_list 和 act_func
        # 后面的元素存储线性层的参数/超参
        self.size_list = param_list[0]
        self.act_func = param_list[1]

        for i in range(len(self.size_list) - 1):
            self.layers = []
            for i in range(len(self.size_list) - 1):
                # 根据维数构建全连接层
                layer = Linear(in_dim=self.size_list[i], out_dim=self.size_list[i + 1])
                # 把全连接层的参数拿出来（从 param_list 的第三个元素开始存储各全连接层的参数）
                layer.W = param_list[i + 2]['W']
                layer.b = param_list[i + 2]['b']
                layer.params['W'] = layer.W
                layer.params['b'] = layer.b
                layer.weight_decay = param_list[i + 2]['weight_decay']
                layer.weight_decay_lambda = param_list[i+2]['lambda']
                # 构建激活函数
                if self.act_func == 'Logistic':
                    raise NotImplemented
                elif self.act_func == 'ReLU':
                    layer_f = ReLU()
                # 当前层放进 self.layers 中
                self.layers.append(layer)
                if i < len(self.size_list) - 2: # 最后一层，不加激活函数
                    self.layers.append(layer_f)


    def save_model(self, save_path):
        param_list = [self.size_list, self.act_func]    # 存储 维数 和 激活函数类型
        for layer in self.layers:
            if layer.optimizable:   # 线性层，再存下一些参数
                param_list.append({'W' : layer.params['W'], 'b' : layer.params['b'], 'weight_decay' : layer.weight_decay, 'lambda' : layer.weight_decay_lambda})
        
        with open(save_path, 'wb') as f:
            pickle.dump(param_list, f)
        

class Model_CNN(Layer):
    """
    A model with conv2D layers. Implement it using the operators you have written in op.py
    """
    def __init__(self, layers=None, dropout_p=1):
        self.layers = layers
        self.p = dropout_p


    def __call__(self, X):
        return self.forward(X)


    def forward(self, X):
        assert self.layers is not None, 'Model has not initialized yet. Use model.load_model to load a model or create a new model with layers offered.'
        outputs = X
        for layer in self.layers:
            outputs = layer(outputs)
            if isinstance(layer, ReLU): # 如果添加新的激活函数，需要在这里加上一点代码
                mask = (np.random.rand(*outputs.shape) < self.p) / self.p
                outputs *= mask
        return outputs


    def backward(self, loss_grad):
        grads = loss_grad
        for layer in reversed(self.layers):
            grads = layer.backward(grads)
        return grads
    

    def predict(self, X):   # 用于训练之外场合的前向传播
        assert self.layers is not None, 'Model has not initialized yet. Use model.load_model to load a model or create a new model with layers offered.'
        outputs = X
        for layer in self.layers:
            outputs = layer(outputs)
        return outputs
    

    def load_model(self, param_list):
        with open(param_list, 'rb') as f:
            param_list = pickle.load(f)
        
        layers = []
        for item in param_list:
            layer_type = item['type']

            if layer_type == 'Linear':
                layer = Linear(
                    in_dim = item['in_dim'],
                    out_dim = item['out_dim'],
                    initialize_method = np.random.normal,
                    weight_decay = item['weight_decay'],
                    weight_decay_lambda = item['weight_decay_lambda']
                )
                layer.params['W'] = item['W']
                layer.params['b'] = item['b']
                layer.W = layer.params['W']
                layer.b = layer.params['b']

            elif layer_type == 'conv2D':
                layer = conv2D(
                    in_channels = item['in_channels'],
                    out_channels = item['out_channels'],
                    kernel_size = item['kernel_size'],
                    stride = item['stride'],
                    padding = item['padding'],
                    initialize_method = np.random.normal,
                    weight_decay = item['weight_decay'],
                    weight_decay_lambda = item['weight_decay_lambda']
                )
                layer.params['W'] = item['W']
                layer.params['b'] = item['b']
                layer.W = layer.params['W']
                layer.b = layer.params['b']
            
            elif layer_type == 'MaxPooling':
                layer = MaxPooling(
                    kernel_size = item['kernel_size'],
                    stride = item['stride'],
                    padding = item['padding']
                )
            
            elif layer_type == 'ReLU':
                layer = ReLU()

            elif layer_type == 'Flatten':
                layer = Flatten()

            else:
                raise ValueError('Undefined layer type')

            layers.append(layer)
            
        self.layers = layers
    
        
    def save_model(self, save_path):
        # param_list 中的一个元素存储一个层的信息
        param_list = []

        # 存储每一层，每一层表示为一个字典
        # 字典中含有：层的类型、创建层所需要的超参数、层中的可优化参数
        for layer in self.layers:

            # 线性层
            if isinstance(layer, Linear):
                param_list.append({
                    'type' : 'Linear',
                    'in_dim' : layer.params['W'].shape[0],
                    'out_dim' : layer.params['W'].shape[1],
                    'weight_decay' : layer.weight_decay, 
                    'weight_decay_lambda' : layer.weight_decay_lambda, 
                    'W' : layer.params['W'], 
                    'b' : layer.params['b'] 
                })
            
            # 卷积层
            elif isinstance(layer, conv2D):
                param_list.append({
                    'type' : 'conv2D',
                    'in_channels': layer.in_channels,
                    'out_channels': layer.out_channels,
                    'kernel_size': layer.kernel_size,
                    'stride': layer.stride,
                    'padding': layer.padding,
                    # 不用存储 initialize_method，反正会被加载的权重覆盖掉
                    'weight_decay': layer.weight_decay,
                    'weight_decay_lambda': layer.weight_decay_lambda,
                    'W': layer.params['W'],
                    'b': layer.params['b']
                })

            # 池化层 MaxPooling
            elif isinstance(layer, MaxPooling):
                param_list.append({
                    'type' : 'MaxPooling',
                    'kernel_size': layer.kernel_size,
                    'stride': layer.stride,
                    'padding': layer.padding
                })
            
            # 激活函数层
            elif isinstance(layer, ReLU):
                param_list.append({
                    'type' : 'ReLU'
                })

            elif isinstance(layer, Flatten):
                param_list.append({
                    'type' : 'Flatten'
                })
            
            else:
                raise ValueError('Undefined layer type')
            

        # 写入文件
        with open(save_path, 'wb') as f:
            pickle.dump(param_list, f)
