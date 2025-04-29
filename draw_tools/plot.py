# plot the score and loss
import matplotlib.pyplot as plt
import numpy as np


colors_set = {'Kraftime' : ('#E3E37D', '#968A62')}

def plot(runner, axes, set=colors_set['Kraftime']):
    train_color = set[0]
    dev_color = set[1]
    
    epochs = [i for i in range(len(runner.train_scores))]

    # 绘制训练损失变化曲线
    axes[0].plot(epochs, runner.train_loss, color=train_color, label="Train loss")
    # 绘制评价损失变化曲线
    axes[0].plot(epochs, runner.dev_loss, color=dev_color, linestyle="--", label="Dev loss")
    # 绘制坐标轴和图例
    axes[0].set_ylabel("loss")
    axes[0].set_xlabel("iteration")
    axes[0].set_title("")
    axes[0].legend(loc='upper right')

    # 绘制训练准确率变化曲线
    axes[1].plot(epochs, runner.train_scores, color=train_color, label="Train accuracy")
    # 绘制评价准确率变化曲线
    axes[1].plot(epochs, runner.dev_scores, color=dev_color, linestyle="--", label="Dev accuracy")
    # 绘制坐标轴和图例
    axes[1].set_ylabel("score")
    axes[1].set_xlabel("iteration")
    axes[1].legend(loc='lower right')



# for CNN
def plot_cnn(runner, axes, set=colors_set['Kraftime']):

    train_color = set[0]
    dev_color = set[1]
    

    # train_loss 的 x 轴坐标
    train_loss_len = len(runner.train_loss)
    train_loss_x = list(range(train_loss_len))
    # 绘制 train_loss 曲线
    axes[0].plot(train_loss_x, runner.train_loss, color=train_color, label="Train loss")

    # dev_loss 的 x 轴坐标
    dev_loss_len = len(runner.dev_loss)
    dev_loss_x = np.linspace(0, train_loss_len - 1, dev_loss_len).tolist()
    # 绘制 dev_loss 曲线
    axes[0].plot(dev_loss_x, runner.dev_loss, color=dev_color, linestyle="--", label="Dev loss")

    # 坐标轴和图例
    axes[0].set_ylabel("loss")
    axes[0].set_xlabel("iteration")
    axes[0].legend(loc='upper right')
    

    # train_scores 的 x 轴坐标
    train_scores_len = len(runner.train_scores)
    train_scores_x = list(range(train_scores_len))
    # 绘制 train_scores 曲线
    axes[1].plot(train_scores_x, runner.train_scores, color=train_color, label="Train accuracy")
    
    # dev_scores 的 x 轴坐标
    dev_scores_len = len(runner.dev_scores)
    dev_scores_x = np.linspace(0, train_scores_len - 1, dev_scores_len).tolist()
    # 绘制 dev_scores 曲线
    axes[1].plot(dev_scores_x, runner.dev_scores, color=dev_color, linestyle="--", label="Dev accuracy")
    
    # 坐标轴和图例
    axes[1].set_ylabel("score")
    axes[1].set_xlabel("iteration")
    axes[1].legend(loc='lower right')
