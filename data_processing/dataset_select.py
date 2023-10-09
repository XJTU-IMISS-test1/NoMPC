import numpy as np
from scipy.sparse import csr_matrix, lil_matrix, coo_matrix

def select_dataset(X, Y, m, m_selected):
    """ 
    对于部分train样本量过大的数据集, 随机抽取其中一部分样本作为训练集 
    m: 原样本量
    m_selected: 抽取之后的预期样本量

    随机抽取seed = 1

    """
    assert X.shape[0] == m
    assert len(Y) == m
    # index_lst = [i for i in range(m)]
    np.random.seed(1)  # 设置随机数种子，保证实验可重复性
    # index = np.random.choice(index_lst, m_selected)  # 抽样后的样本序号
    index = np.random.choice(m, m_selected, replace=False)    # 抽取不重复的序号需要设置replace参数为False
    index.sort()  # 序号由小到大排列
    assert len(index) == m_selected  # 检查长度一致性
    X_selected = X[index, :]
    Y_selected = Y[index]
    return X_selected, Y_selected

def load_dataset(dataset_name):
    # return X, Y
    pass

def dataset_selector(dataset_name, train_data, test_data, Y_train):
    """ 
    function 1.根据不同数据集, 选取抽取的样本数量, 将抽取的结果返回 
    function 2.对于某些稀疏数据集, 扩充数据特征维度(这些返回的是正常的数据集, 不需要再取了)

    这里统一不转成dense, 就是稀疏, 后面有需要再转

    dataset_name: 数据集名称

    train_data: 稀疏矩阵格式的训练集数据, 包含了特征和标签

    Y_train: 从train_data提取出的标签

    keep_sparse: 对于高维稀疏的超大数据集, 保持稀疏格式, 否则可能超内存

    返回: 均是可以使用的sparse格式训练特征和标签
    """
    X_train = train_data[0]
    X_test = test_data[0]

    if dataset_name == "cifar10":
        X_train = train_data[0].todense().A
        X_train, Y_train = select_dataset(X_train, Y_train, 50000, 10000)
    elif dataset_name == "SVHN":
        X_train = train_data[0].todense().A
        X_train, Y_train = select_dataset(X_train, Y_train, 34750, 10000)
        # raise NotImplementedError
    elif dataset_name == "webspam10k_50k":
        X_train = csr_matrix(X_train, shape=(X_train.shape[0], 50000))
        X_test = csr_matrix(X_test, shape=(X_test.shape[0], 50000))

    elif dataset_name == "webspam10k_100k":
        X_train = csr_matrix(X_train, shape=(X_train.shape[0], 100000))
        X_test = csr_matrix(X_test, shape=(X_test.shape[0], 100000))

    elif dataset_name == "webspam10k_500k":
        X_train = csr_matrix(X_train, shape=(X_train.shape[0], 500000))
        X_test = csr_matrix(X_test, shape=(X_test.shape[0], 500000))

    # elif keep_sparse == True:
    #     return X_train, X_test, Y_train
    # else:
    #     # 不需要抽取的数据集, 直接返回结果, 目前返回的X_train还是稀疏矩阵格式, Y_train是numpy数组
    #     # 现在返回的都是numpy
    #     X_train = train_data[0].todense().A
    #     Y_train = Y_train

    return X_train, X_test, Y_train



if __name__ == '__main__':
    dataset_name = "cifar10"
    X, Y = load_dataset(dataset_name)

    if dataset_name == "cifar10":
        """ Train: (50000, 3072) Test: (50000,) """
        select_dataset(X, Y, 50000, 10000)