import scipy.sparse as sp
import numpy as np
from sklearn.datasets import load_svmlight_file
import os

def load_txt(path, loadtype):
    return np.loadtxt(path, delimiter=',', dtype = loadtype)
def save_txt(path, data, format):
    np.savetxt(path, data, delimiter=',', fmt = format)

def samples_to_sketch_PMinhash(m, n, k, samples):  # m个样本，n个特征
    assert m == samples.shape[0]  # 若samples的样本数与原数据样本数不一致，触发异常
    assert k == samples.shape[1]  # 采样次数k与samples的列数不一致，则触发异常
    X = sp.lil_matrix((m, n * k), dtype=int)  # lil_matrix（适合用于替换值）数据为int型的稀疏矩阵，行列数与原数据集的稀疏矩阵保持一致

    # @@@注意：原数据的值全为0，(以n=6,k=4距离)即x=[0,0,0,0,0,0]，CWS得到的采样samples=[-1,-1,-1,-1]，则sketch的one-hot编码全为0
    # 遍历(m,k)的sketch二维数组
    # 统计sample=-1出现的次数
    sum_zero_sample = 0
    for i in range(m):
        for j in range(k):
            sample_value = samples[i, j]
            if(sample_value == -1):
                sum_zero_sample += 1
                continue
            assert sample_value >= 1
            index = n * j + n - sample_value  # 一个one-hot编码中唯一的1对应的标号：例如对j=1（对应cws的第一个sample:i1），对应one-hot编码中为1的位置：n - i1
            X[i, index] = 1
            # print("\rRemain:{}".format(m-i), end = '')
    print('采样为0的情况出现的次数: ', sum_zero_sample)
    return X

def sampling_data_2_encoded_data(main_path, dataset_name, 
            train_file_name1, train_file_name2, feature_total_num, sampling_num):
    '''
    Description: 采样得到的结果 sampling result 转化为 encoded data

    Parameters
    ---

    数据路径: main_path, dataset_name, train_file_name1, train_file_name2

    encoding参数: 特征总量-feature_total_num, 采样次数-sampling_num
    '''

    # 分片3的sampling数据
    samples_3 = load_txt(os.path.join(main_path, dataset_name, train_file_name1), 'int')
    # X1_train_sketch - 3:7-3  1000样本 307采样 18特征
    # X2_.. - 3:7-7
    print(samples_3.shape)
    # 分片7的sampling数据
    samples_7 = load_txt(os.path.join(main_path, dataset_name, train_file_name2), 'int')
    print(samples_7.shape, type(samples_7))


    # encoding参数设置
    # m = 1000 # 样本数
    # n = 60 # 特征数
    # k = 1024 # 采样次数
    partition = 3/10

    m1, m2 = samples_3.shape[0], samples_7.shape[0]
    # print(samples_3.shape[0], samples_7.shape[0],samples_3.shape[1], samples_7.shape[1]) # 1000 1000 307 717
    assert m1 == m2
    m = m1 # 设置样本数
    assert sampling_num == samples_3.shape[1] + samples_7.shape[1]
    n = feature_total_num # 设置特征数
    k = sampling_num # 设置采样次数

    print("样本数为:{},特征数为:{},采样次数为:{}".format(m, n, k))


    # n1 = np.floor(n * partition).astype(int)        # 第一部分数据集特征维度
    # n2 = n - n1                                   # 第二部分数据集特征维度
    k1 = np.floor(k * partition).astype(int)
    k2 = k - k1
    print("sampling num k1:{}, k2:{}".format(k1,k2))


    # 分片拼接
    X_train_sampling = np.c_[samples_3, samples_7]
    print(X_train_sampling.shape, type(X_train_sampling), type(X_train_sampling[0][0]))
    X_train_sampling_int = X_train_sampling.astype(np.int)
    print(X_train_sampling_int.shape, type(X_train_sampling_int), type(X_train_sampling_int[0][0]))

    # encoding 得到最终的数据集
    result_X = samples_to_sketch_PMinhash(m, n, k, X_train_sampling_int)
    print(result_X.shape, type(result_X))

    print(n * k1)
    # 将最终数据集拆分成两方的数据集
    result_X_dense = result_X.todense()
    print(result_X_dense.shape)
    result_X_train1, result_X_train2 = result_X_dense[:,0:n*k1], result_X_dense[:,n*k1:] # np.hsplit(result_X_dense, [0, n * k1])
    print(result_X_train1.shape, result_X_train2.shape)

    return result_X_train1, result_X_train2
    # np.savetxt("/Users/zbz/code/vscodemac_python/hetero_sshe_logistic_regression/data/splice/distrubuted/X1_encoded.txt", result_X_train1, delimiter=',') #, fmt='%.2f')
    # np.savetxt("/Users/zbz/code/vscodemac_python/hetero_sshe_logistic_regression/data/splice/distrubuted/X2_encoded.txt", result_X_train2, delimiter=',') #, fmt='%.2f')
    # print("saving...Done.")


def sampling_data_2_encoded_data_testdataset(main_path, dataset_name, 
            train_file_name1, train_file_name2, feature_total_num, sampling_num):
    '''
    Description: 采样得到的结果 sampling result 转化为 encoded data

    Parameters
    ---

    数据路径: main_path, dataset_name, train_file_name1, train_file_name2

    encoding参数: 特征总量-feature_total_num, 采样次数-sampling_num
    '''

    # 分片3的sampling数据
    samples_3 = load_txt(os.path.join(main_path, dataset_name, train_file_name1), 'int')
    # X1_train_sketch - 3:7-3  1000样本 307采样 18特征
    # X2_.. - 3:7-7
    print(samples_3.shape)
    # 分片7的sampling数据
    samples_7 = load_txt(os.path.join(main_path, dataset_name, train_file_name2), 'int')
    print(samples_7.shape, type(samples_7))


    # encoding参数设置
    # m = 1000 # 样本数
    # n = 60 # 特征数
    # k = 1024 # 采样次数
    partition = 3/10

    m1, m2 = samples_3.shape[0], samples_7.shape[0]
    # print(samples_3.shape[0], samples_7.shape[0],samples_3.shape[1], samples_7.shape[1]) # 1000 1000 307 717
    assert m1 == m2
    m = m1 # 设置样本数
    assert sampling_num == samples_3.shape[1] + samples_7.shape[1]
    n = feature_total_num # 设置特征数
    k = sampling_num # 设置采样次数

    print("样本数为:{},特征数为:{},采样次数为:{}".format(m, n, k))


    # n1 = np.floor(n * partition).astype(int)        # 第一部分数据集特征维度
    # n2 = n - n1                                   # 第二部分数据集特征维度
    k1 = np.floor(k * partition).astype(int)
    k2 = k - k1
    print("sampling num k1:{}, k2:{}".format(k1,k2))


    # 分片拼接
    X_train_sampling = np.c_[samples_3, samples_7]
    print(X_train_sampling.shape, type(X_train_sampling), type(X_train_sampling[0][0]))
    X_train_sampling_int = X_train_sampling.astype(np.int)
    print(X_train_sampling_int.shape, type(X_train_sampling_int), type(X_train_sampling_int[0][0]))

    # encoding 得到最终的数据集
    result_X = samples_to_sketch_PMinhash(m, n, k, X_train_sampling_int)
    print(result_X.shape, type(result_X))

    result_X_dense = result_X.todense()

    return result_X_dense


if __name__ == '__main__':

    
    main_path = '/Users/zbz/code/vscodemac_python/hetero_sshe_logistic_regression/data/'
    dataset_name = 'splice/distrubuted/'
    train_file_name1 = 'X1_train_sketch.txt'
    train_file_name2 = 'X2_train_sketch.txt'
    
    # result_X_train1, result_X_train2 = sampling_data_2_encoded_data(main_path, dataset_name, train_file_name1, train_file_name2, 
    #         feature_total_num = 60, sampling_num = 1024)
    # np.savetxt("/Users/zbz/code/vscodemac_python/hetero_sshe_logistic_regression/data/splice/distrubuted/encoded/X1_encoded_train37.txt", result_X_train1, delimiter=',') #, fmt='%.2f')
    # np.savetxt("/Users/zbz/code/vscodemac_python/hetero_sshe_logistic_regression/data/splice/distrubuted/encoded/X2_encoded_train37.txt", result_X_train2, delimiter=',') #, fmt='%.2f')

    result_X_train = sampling_data_2_encoded_data_testdataset(main_path, dataset_name, train_file_name1, train_file_name2, 
            feature_total_num = 60, sampling_num = 1024)
    np.savetxt("/Users/zbz/code/vscodemac_python/hetero_sshe_logistic_regression/data/splice/distrubuted/encoded/X_encoded_train37.txt", result_X_train, delimiter=',') #, fmt='%.2f')

    train_file_name1 = 'X1_test_sketch.txt'
    train_file_name2 = 'X2_test_sketch.txt'

    # result_X_train1, result_X_train2 = sampling_data_2_encoded_data(main_path, dataset_name, train_file_name1, train_file_name2, 
    #         feature_total_num = 60, sampling_num = 1024)
    # np.savetxt("/Users/zbz/code/vscodemac_python/hetero_sshe_logistic_regression/data/splice/distrubuted/encoded/X1_encoded_test37.txt", result_X_train1, delimiter=',', fmt='%i') #, fmt='%.2f')
    # np.savetxt("/Users/zbz/code/vscodemac_python/hetero_sshe_logistic_regression/data/splice/distrubuted/encoded/X2_encoded_test37.txt", result_X_train2, delimiter=',', fmt='%i') #, fmt='%.2f')

    result_X_test = sampling_data_2_encoded_data_testdataset(main_path, dataset_name, train_file_name1, train_file_name2, 
            feature_total_num = 60, sampling_num = 1024)
    np.savetxt("/Users/zbz/code/vscodemac_python/hetero_sshe_logistic_regression/data/splice/distrubuted/encoded/X_encoded_test37.txt", result_X_test, delimiter=',', fmt='%i') #, fmt='%.2f')
    