#! ~/opt/anaconda3/bin/python3.8
from model import LogisticRegression
from data_preprocess import *
from matplotlib import pyplot as plt

"""
LR含义是逻辑回归
"""
# 超参数，请自行实验并设置
LR =  1e-2    # 学习率
BATCH_SIZE = 8
EPOCHS = 10

# data_preprocess生成的数据集路径：
train_set_path = './train_set.pkl'
test_set_path = './test_set.pkl'


def draw_loss(train_losses, test_losses):
    """
    绘制损失曲线图
    :param train_losses: 训练损失
    :param test_losses: 测试损失
    :return:
    """
    epochs = range(1, len(train_losses)+1)
    plt.plot(epochs, train_losses, 'r', label='Train Loss')
    plt.plot(epochs, test_losses, 'b', label='Test Loss')
    plt.legend()

    plt.title('Training and Test Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.show()
def draw_acc(train_losses, acc):
    epochs = range(1, len(train_losses) + 1)
    plt.plot(epochs, acc, 'r', label='Test Acc')
    plt.legend()

    plt.title('Test Acc')
    plt.xlabel('Epochs')
    plt.ylabel('Acc')
    plt.show()

if __name__ == "__main__":
    # 加载训练集：
    with open(train_set_path, 'rb') as f:
        train_set = pickle.load(f)

    # 加载测试集：
    with open(test_set_path, 'rb') as f:
        test_set = pickle.load(f)
    # 制作迭代器
    train_iter = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, drop_last=False)

    # 建立模型
    model = LogisticRegression(epochs=EPOCHS, learning_rate=LR)

    # 训练模型并记录损失
    train_losses, test_losses, acc = model.train(train_iter, test_set)

    # 测试模型精度
    accuracy = model.test_accuracy(test_set)
    a = np.array(train_losses)
    # print(a.shape)
    print("accuracy:", accuracy, "%")

    # 绘制损失曲线
    draw_loss(train_losses, test_losses)
    draw_acc(train_losses, acc)




