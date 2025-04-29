import mynn as nn
import numpy as np
from struct import unpack
import gzip
import matplotlib.pyplot as plt
import pickle

model = nn.models.Model_CNN()
model.load_model(r'.\cnn_models\best_model_9720.pickle')

test_images_path = r'.\dataset\MNIST\t10k-images-idx3-ubyte.gz'
test_labels_path = r'.\dataset\MNIST\t10k-labels-idx1-ubyte.gz'

with gzip.open(test_images_path, 'rb') as f:
        magic, num, rows, cols = unpack('>4I', f.read(16))
        test_imgs=np.frombuffer(f.read(), dtype=np.uint8).reshape(num, 28*28)
    
with gzip.open(test_labels_path, 'rb') as f:
        magic, num = unpack('>2I', f.read(8))
        test_labs = np.frombuffer(f.read(), dtype=np.uint8)


# test_num = 10000
# test_imgs = test_imgs[:test_num]
# test_labs = test_labs[:test_num]
N_test = test_imgs.shape[0]
# CNN 需要转换输入尺寸
test_imgs = test_imgs.reshape(N_test, 1, 28, 28)

test_imgs = test_imgs / test_imgs.max() # 归一化

logits = model.predict(test_imgs)       # 调用 model，获取预测概率
print(nn.metric.accuracy(logits, test_labs))