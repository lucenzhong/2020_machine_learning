import numpy as np
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
import pandas as pd
import copy
import matplotlib.pyplot as plt

'''
实现线性回归训练和推理，要求如下：

1、训练过程是一个函数，该函数至少包括三个输入参数，训练实例集、训练实例的标签和学习率，采用梯度下降法训练。
2、推理过程是一个函数。
3、语言不限，推荐C++或者Python。
4、需要包括一个验证可以验证训练过程和推理过程的完整调用示例。

'''
# 程序运行方式： python linear_regression.py
# 数据集：波士顿房价数据集   数据集介绍：https://www.kesci.com/home/dataset/590bd595812ede32b73f55f2
# 如果运行有问题，请学长/学姐随时联系我 qq:1406997434


def save_data_as_csv():
    """
    将数据保存为本地housing.csv, 避免重复从sklearn中加载数据

    """
    data, target = load_boston(return_X_y=True)
    target = target.reshape(-1, 1)
    dataset = np.hstack((data, target))
    np.savetxt("housing.csv", dataset)


def preprocess_data():
    """
    1、进行特征选择
    2、数据正规化处理
    3、划分训练集和测试集

    :return: train_x, test_x, train_y, test_y
    """
    data = pd.read_csv('./housing.csv', header=None, sep='\s+')
    data.columns = ['CRIM', 'ZN', 'INDUS', 'CHAS',
                      'NOX', 'RM', 'AGE', 'DIS', 'RAD',
                      'TAX', 'PTRATIO', 'B', 'LSTAT', 'Price']   # 预测的目标为房价Price(最后一列)

    # print(data.corr()['Price']) =>
    # 输出结果为：
    # CRIM - 0.388305 ZN 0.360445 INDUS - 0.483725 CHAS 0.175260 NOX - 0.427321 RM 0.695360
    # AGE - 0.376955 DIS 0.249929 RAD - 0.381626 TAX - 0.468536 PTRATIO - 0.507787 B 0.333461 LSTAT - 0.737663
    # 和Price相关系数大于0.5的特征有3个： RM PTRATIO LSTAT

    # 特征选择
    x = np.array(data[['RM', 'PTRATIO', 'LSTAT']])
    y = np.array(data['Price'])

    # 对数据进行正规化处理
    mean = np.mean(x, axis=0)
    std = np.std(x, axis=0)
    x_norm = (x - mean) / std

    # 划分训练集和测试集，本次实验中，用几个数据（2%）进行验证（推断）即可
    train_x, test_x, train_y, test_y = train_test_split(x_norm, y, test_size=0.02)

    return train_x, test_x, train_y, test_y


def train(train_x, train_y, lr, epoch=100, lamda=1):
    """
    训练一个多元线性回归模型，并加入正则化项防止过拟合

    :param train_x: 训练数据
    :param train_y: 目标值
    :param lr: 学习率
    :param epoch: 训练轮次，默认为100   =>可根据训练过程中loss曲线动态选择
    :param lamda: 正则化系数，默认为1
    :return: theta 训练后的线性方程系数
    """
    m = len(train_y)   # m为训练样本数量
    train_errors = []   # 保存每一个epoch的损失函数值，用于画loss曲线
    train_x = np.column_stack((np.ones(m), train_x))   # 在训练数据中增加一列x0=1
    theta = np.zeros(train_x.shape[1])    # 初始化线性方程的系数为[0, 0, 0, 0]

    for i in range(1, epoch):
        theta_r = copy.copy(theta)
        theta_r[0] = 0
        # 使用BGD计算梯度  =>梯度和损失函数的表达式和课件04_Training_Models中加入正则化项后的表达式一致
        theta = theta - lr / m * (np.matmul(train_x.T, np.matmul(train_x, theta) - train_y) +
                                  lamda/m * theta_r)
        loss = 1/(2*m) * np.matmul((np.matmul(train_x, theta) - train_y).T,
                                   np.matmul(train_x, theta) - train_y) + \
               lamda / (2*m) * np.matmul(theta_r.T, theta_r)
        train_errors.append(loss)

    # 以下四行代码可以用于画训练过程的损失函数值，plt.show()之后需要关掉图片后才可以继续往下运行
    # plt.plot(train_errors, "r-+", linewidth=2, label="train")
    # plt.xlabel("Epoch", fontsize=14)
    # plt.ylabel("Training loss", fontsize=14)
    # plt.show()

    return theta


def inference(test_x, test_y, theta, lamda=1):
    """
    在测试集上验证训练完成的多元线性回归模型

    :param test_x: 测试数据
    :param test_y: 目标值
    :param theta: 训练得到的系数
    :param lamda: 正则化系数，默认为1
    """
    m = len(test_y)
    test_x = np.column_stack((np.ones(m), test_x))
    theta_r = copy.copy(theta)
    theta_r[0] = 0
    # 计算测试集上损失函数的值
    loss = 1 / (2 * m) * np.matmul((np.matmul(test_x, theta) - test_y).T,
                                   np.matmul(test_x, theta) - test_y) + \
           lamda/(2*m) * np.matmul(theta_r.T, theta_r)
    print('测试集上的loss为：', loss)

    # 绘制推理结果
    plt.plot(np.matmul(test_x, theta),'+-', label="predict price")
    plt.plot(test_y, '*-', label="True price")
    plt.legend(loc="upper right", fontsize=10)
    plt.xlabel('test data', fontsize=14)
    plt.ylabel('price', fontsize=14)
    plt.show()


def main():
    # 加载数据
    save_data_as_csv()    # 只需要运行一次将数据保存到本地即可
    # 数据预处理
    train_x, test_x, train_y, test_y = preprocess_data()
    # 模型训练
    theta = train(train_x, train_y, lr=0.05)    # 得到回归系数后，若有新的数据x=[1, a, b, c]，直接与theta进行矩阵乘法即可得到预测的目标值
    # 模型验证、推理
    inference(test_x, test_y, theta)


if __name__ == '__main__':
    main()