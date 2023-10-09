import numpy as np
import time
from sklearn.kernel_approximation import PolynomialCountSketch, Nystroem, RBFSampler

'''
求数据集-稀疏矩阵的方差
'''
def cal_X_var(X):
    X_mean = X.mean()  # 矩阵X的均值——全部值求平均
    if isinstance(X, np.ndarray):
        X_square = np.multiply(X, X)  # 矩阵X平方, 点对点元素相乘
    else:
        X_square = X.multiply(X)
    
    X_var = X_square.mean() - X_mean ** 2  # 方差Var = x平方均值 - x均值平方 (= E(X^2) - E(X)^2)

    return X_var

def dataSketch_generator_RBFandPolyKernel(X_train, X_test, Y_train, Y_test, kernel_method, sampling_k, partition):
    """
    使用 RBF和Poly两种核的近似方法 —— RFF和TensorSketch, 处理数据集
    返回处理后的数据集和近似方法的参数: gamma_scale
    """
    k = int(sampling_k)
    k1 = np.floor(k * partition).astype(int)
    k2 = k - k1

    # X_train
    # partition = 3/10
    n_train = X_train.shape[1] # 总特征数
    n1 = np.floor(n_train * partition).astype(int) # X1的特征数
    n2 = n_train - n1
    X_train1, X_train2 = X_train[:,0:n1], X_train[:,n1:]    # X_train1, X_train2 = X_train[:,0:n1], X_train[:,n2:] ...写错了

    # X_test
    # partition = 3/10
    n_test = X_test.shape[1] # 总特征数
    n1 = np.floor(n_test * partition).astype(int)
    n2 = n_test - n1
    X_test1, X_test2 = X_test[:,0:n1], X_test[:,n1:]


    gamma_scale = 1. / X_train.shape[1] / cal_X_var(X_train)
    # gamma_scale = 1

    if kernel_method == "rff":
        rff1 = RBFSampler(gamma=gamma_scale, n_components=k1, random_state=1) # random_state随机数种子
        rff2 = RBFSampler(gamma=gamma_scale, n_components=k2, random_state=1) # random_state随机数种子
        # 生成sketch
        X1_train_sketch = rff1.fit_transform(X_train1)
        X1_test_sketch = rff1.fit_transform(X_test1)
        X2_train_sketch = rff2.fit_transform(X_train2)
        X2_test_sketch = rff2.fit_transform(X_test2)

    elif kernel_method == "poly":
        ts1 = PolynomialCountSketch(degree=2, gamma=gamma_scale, coef0=1, n_components=k1, random_state=1)
        ts2 = PolynomialCountSketch(degree=2, gamma=gamma_scale, coef0=1, n_components=k2, random_state=1)

        # 生成skech
        X1_train_sketch = ts1.fit_transform(X_train1)
        X1_test_sketch = ts1.fit_transform(X_test1)
        X2_train_sketch = ts2.fit_transform(X_train2)
        X2_test_sketch = ts2.fit_transform(X_test2)
    
    print("data generation ({}) ok.".format(kernel_method))

    return X1_train_sketch, X2_train_sketch, Y_train, X1_test_sketch, X2_test_sketch, Y_test, gamma_scale

if __name__ == "__main__":
    pass