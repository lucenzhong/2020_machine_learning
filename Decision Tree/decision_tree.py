from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import treePlotter

'''
实现决策树训练和推理，要求如下：
1、训练过程是一个函数，该函数至少包括四个输入参数，训练实例集、训练实例的标签、最大深度和叶子结点最少样本数，Gini或者熵训练都可以。
2、推理过程是一个函数。
3、语言不限，推荐C++或者Python。
4、需要包括一个验证可以验证训练过程和推理过程的完整调用示例。

实验说明：
1.程序运行方式：python decision_tree.py
2.程序输出：正确标签、预测标签、准确率、图形化的决策树
3.tree.Plotter.py文件仅仅是为了能更直观的看到训练结果，直接用网上比较常见的画决策树的代码做了一点修改，不是自己写的
4.选用iris数据集，实现了属性值连续的决策树算法。所有计算公式均参考周志华《机器学习》第四章决策树内容
5.如果运行出现问题，请学长/学姐随时联系我。 qq:1406997434

'''

depth_count = 0   # 记录当前计算结点的深度


class Node(object):
    """
    存储一棵树的结点的所有信息
    """
    def __init__(self):
        self.feature_name = None   # 属性名称
        self.subtree = {}   # 子树，用字典形式存储决策树
        self.impurity = None
        self.split_value = None   # 切分值
        self.is_leaf = False   # 是否为叶子结点，True表示是，False表示否
        self.leaf_class = None   # 叶子节点的类别
        self.leaf_num = None   # 叶子节点的格式
        self.high = -1   # 结点的深度
        self.num_sample = 0   # 节点包含的样本数


def process_data():
    """
    1.加载数据集
    2.x转为pd.DataFrame数据类型，y转为pd.Series数据类型，方便用pandas进行数据分析
    :return: x_train, x_test, y_train, y_test
    """
    dataset = load_iris()
    x = dataset.data
    y = dataset.target
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
    x_train = pd.DataFrame(x_train, columns=['SepalLength', 'SepalWidth', 'PetalLength', 'PetalWidth'])
    y_train = pd.Series(y_train)
    x_test = pd.DataFrame(x_test, columns=['SepalLength', 'SepalWidth', 'PetalLength', 'PetalWidth'])
    y_test = pd.Series(y_test)
    # print('训练集中数据类别分析结果：\n', pd.value_counts(y_train))
    return x_train, x_test, y_train, y_test


def tree_generate(x_train, y_train, max_depth, mini_sample):
    """
    生成决策树
    :param x_train: 训练实例集
    :param y_train: 训练实例的标签
    :param max_depth: 最大深度
    :param mini_sample: 叶子结点最少样本数
    :return: 树
    """
    global depth_count
    # 对树进行初始化
    my_tree = Node()
    my_tree.leaf_num = 0
    my_tree.num_sample = len(y_train.values)

    # 如果当前计算的结点深度等于最大深度，则将其设置为叶节点，返回样本数最多的类
    if depth_count == max_depth:
        my_tree.is_leaf = True
        my_tree.leaf_class = y_train.values[0]
        my_tree.high = 0
        my_tree.leaf_num += 1
        return my_tree

    # 如果当前结点的数据属于同一类别，则将其设置为叶节点，并返回这个类
    if y_train.nunique() == 1:
        my_tree.is_leaf = True
        my_tree.leaf_class = y_train.values[0]
        my_tree.high = 0
        my_tree.leaf_num += 1
        return my_tree

    best_feature, best_impurity = choose_best_feature(x_train, y_train)   # 从特征中选择最优划分属性
    # 记录结点信息
    my_tree.feature_name = best_feature
    my_tree.impurity = best_impurity[0]
    my_tree.split_value = best_impurity[1]
    feature_values = x_train.loc[:, best_feature]   # 最优划分属性的值

    # 定义划分出的两颗子树的key(键值)
    up_part = '≥{:.2f}'.format(my_tree.split_value)
    down_part = '<{:.2f}'.format(my_tree.split_value)

    depth_count += 1   # 进行划分，数的深度+1
    # 如果划分后即为最大深度，但是划分后结果为同一类，则停止划分
    if pd.value_counts(y_train[feature_values >= my_tree.split_value]).index[0] == pd.value_counts(y_train[feature_values < my_tree.split_value]).index[0] and depth_count == max_depth:
        my_tree.is_leaf = True
        my_tree.leaf_class = pd.value_counts(y_train).index[0]
        my_tree.high = 0
        my_tree.leaf_num += 1
        return my_tree

    # 若划分后叶子结点的样本数小于最小样本数，则停止划分
    if len(y_train[feature_values >= my_tree.split_value].values) < mini_sample or len(y_train[feature_values < my_tree.split_value].values) < mini_sample:
        my_tree.is_leaf = True
        my_tree.leaf_class = pd.value_counts(y_train).index[0]
        my_tree.high = 0
        my_tree.leaf_num += 1
        return my_tree

    # 以上条件均不满足，才可以进行划分
    my_tree.subtree[up_part] = tree_generate(x_train[feature_values >= my_tree.split_value],
                                             y_train[feature_values >= my_tree.split_value], max_depth, mini_sample)
    my_tree.subtree[down_part] = tree_generate(x_train[feature_values < my_tree.split_value],
                                               y_train[feature_values < my_tree.split_value], max_depth, mini_sample)

    # 记录叶子结点个数
    my_tree.leaf_num += (my_tree.subtree[up_part].leaf_num + my_tree.subtree[down_part].leaf_num)

    # 记录数的深度
    my_tree.high = max(my_tree.subtree[up_part].high, my_tree.subtree[down_part].high) + 1

    return my_tree


def choose_best_feature(x_train, y_train):
    """
    选择当前最优的划分属性
    :param x_train: 训练实例集
    :param y_train: 训练实例的标签
    :return: 最优属性, [最小基尼系数， 最佳划分点]
    """
    features = x_train.columns   # 候选属性集
    best_feature = None   # 最佳划分属性
    best_gini = [float('inf')]   # 最小基尼系数

    # 分别求出每一个划分属性对应的基尼系数，并保存最小的
    for feature_name in features:
        gini_indx = gini_index(x_train[feature_name], y_train)
        if gini_indx[0] < best_gini[0]:
            best_feature = feature_name
            best_gini = gini_indx

    return best_feature, best_gini


def gini_index(feature, y):
    """
    选择基尼系数最小的点作为该特征的分割点
    :param feature: 特征
    :param y: 类别值
    :return:基尼系数，最佳划分点
    """
    m = y.shape[0]   # 结点的样本数
    unique_value = pd.unique(feature)   # 选出feature中出现的不同取值
    unique_value.sort()   # 对取值进行排序
    split_point_set = [(unique_value[i] + unique_value[i+1]) / 2
                       for i in range(len(unique_value) - 1)]   # 获得划分点集合
    mini_gini = float('inf')
    mini_gini_point = None

    # 寻找该特征的最优划分点，使基尼系数最小
    for split_point in split_point_set:
        # 划分点将数据分为两类
        Dv1 = y[feature <= split_point]
        Dv2 = y[feature > split_point]
        gini_index = Dv1.shape[0] / m * gini(Dv1) + Dv2.shape[0] / m * gini(Dv2)   # 计算基尼系数
        # 保存基尼系数最小的划分点
        if gini_index < mini_gini:
            mini_gini = gini_index
            mini_gini_point = split_point

    return mini_gini, mini_gini_point


def gini(y):
    """
    算划分后样例集合的基尼系数
    """
    p = pd.value_counts(y) / y.shape[0]
    gini = 1 - np.sum(p ** 2)   # 在划分后的样本中任意选出两个，标签不同的概率
    return gini


def evaluate(x_test, y_test, my_tree):
    """
    1.实现对于新的样本的标签推断
    2.评估算法的准确率
    :param x_test: 测试实例集
    :param y_test: 测试实例的标签
    :param my_tree: 训练好的决策树
    :return: 算法准确率
    """
    m = y_test.shape[0]
    y_predict = []   # 保存推断结果
    for i in range(m):
        y_predict.append(inference(x_test[i:i+1], my_tree))
    y_predict = pd.Series(y_predict)
    print('正确标签: ', y_test.values)
    print('预测标签: ', y_predict.values)
    precision = (y_test == y_predict).value_counts(normalize=True)[True]   # 比较原始标签和预测标签，得到准确率

    return precision


def inference(x_test, subtree):
    """
    对单一样本进行推断
    :param x_test: 单一样本数据
    :param subtree: 决策树
    :return: 预测的样本标签
    """
    # 如果递归到了叶子结点，就返回改叶子结点对于的标签
    if subtree.is_leaf:
        return subtree.leaf_class

    # 比较样本数据的值和划分点的值，递归进行推断
    if x_test[subtree.feature_name].values >= subtree.split_value:
        return inference(x_test, subtree.subtree['≥{:.2f}'.format(subtree.split_value)])
    else:
        return inference(x_test, subtree.subtree['<{:.2f}'.format(subtree.split_value)])


def main():
    x_train, x_test, y_train, y_test = process_data()   # 划分数据集
    my_tree = tree_generate(x_train, y_train, max_depth=4, mini_sample=10)   # 构造决策树
    precision = evaluate(x_test, y_test, my_tree)   # 进行推断和评估
    print('准确率 = ', precision)
    treePlotter.create_plot(my_tree)   # 画出训练的决策树，便于观察结果


if __name__ == '__main__':
    main()