import numpy as np
import scipy.sparse as sp
import os
# 分布式
def Dis_samples_to_sketch(m, n, n1, k, b, c, samples1, samples2):
    """
    :param m: 样本数
    :param n: 特征数
    :param n1: 特征划分的第一部分数据集的最大特征数
    :param k: P-minhash和0 bit CWS的采样次数（采样结果samples1，samples2的列数之和）
    :param b: b bit minwise hash的b（将特征数n压缩至2^b）
    :param c: Count Sketch的c（将特征数n压缩至c）
    :param samples1: 特征划分的第一部分数据集的CWS采样结果
    :param samples2: 特征划分的第二部分数据集的CWS采样结果
    :return: 将P-minhash和0 bit CWS得到的sample转为sketch
    """

    assert m == samples1.shape[0]                        # 若samples的样本数与原数据样本数不一致，触发异常
    assert m == samples2.shape[0]
    assert k == samples1.shape[1] + samples2.shape[1]    # 采样次数k与samples的列数不一致，则触发异常
    # 1 不进行任何特征压缩
    if (b == 0 and c == 0):
        X = sp.lil_matrix((m, n * k), dtype=int)  # lil_matrix（适合用于替换值）数据为int型的稀疏矩阵，行列数与原数据集的稀疏矩阵保持一致
        sum_zero_sample1 = 0  # 统计sample=0出现的次数
        sum_zero_sample2 = 0  # 统计sample=0出现的次数
        sum_zero_instance1 = 0  # 统计全0instance
        sum_zero_instance2 = 0  # 统计全0instance
        for i in range(m):   # 遍历(m,k)的sketch二维数组
            if (samples1[i, 0] == -1 or samples1[i, 0] == 0):
                sum_zero_instance1 += 1
            if (samples2[i, 0] == -1 or samples2[i, 0] == 0):
                sum_zero_instance2 += 1
            for j in range(samples1.shape[1]):
                sample_value1 = samples1[i, j]
                if (sample_value1 == -1 or sample_value1 == 0):
                    sum_zero_sample1 += 1
                    continue
                assert sample_value1 >= 1
                index = n * j + n - sample_value1  # 一个one-hot编码中唯一的1对应的标号：例如对j=1（对应cws的第一个sample:i1），对应one-hot编码中为1的位置：n - i1
                X[i, index] = 1
            for t in range(samples1.shape[1], k):
                sample_value2 = samples2[i, t - samples1.shape[1]]
                if (sample_value2 == -1 or sample_value2 == 0):
                    sum_zero_sample2 += 1
                    continue
                assert sample_value2 >= 1
                index = n * t + n - (sample_value2 + n1)   # 一个one-hot编码中唯一的1对应的标号：例如对j=1（对应cws的第一个sample:i1），对应one-hot编码中为1的位置：n - i1
                X[i, index] = 1

        print('采样为0的情况出现的次数: ', sum_zero_sample1, sum_zero_sample2)
        print('全0instance: ', sum_zero_instance1, sum_zero_instance2)
        X = X.tocsr()
        return X
    # 2 进行Count Sketch特征压缩
    elif (b == 0 and c != 0):
        X = sp.lil_matrix((m, c * k), dtype=int)
        sum_zero_sample1 = 0  # 统计sample=0出现的次数
        sum_zero_sample2 = 0  # 统计sample=0出现的次数
        sum_zero_instance1 = 0  # 统计全0instance
        sum_zero_instance2 = 0  # 统计全0instance
        # 创建随机数组用于构成哈希函数h1~hk
        np.random.seed(1)
        a_lst = np.random.randint(low=1, high=c, size=k)
        b_lst = np.random.randint(low=0, high=c, size=k)
        for i in range(m):
            if (samples1[i, 0] == -1 or samples1[i, 0] == 0):
                sum_zero_instance1 += 1
            if (samples2[i, 0] == -1 or samples2[i, 0] == 0):
                sum_zero_instance2 += 1
            for j in range(samples1.shape[1]):
                sample_value1 = samples1[i, j]
                if (sample_value1 == -1 or sample_value1 == 0):
                    sum_zero_sample1 += 1
                    continue
                assert sample_value1 >= 1
                hash_value1 = (a_lst[j] * sample_value1 + b_lst[j]) % c + 1   # 经过Count Sketch哈希，将数据从[1,n]映射成了[0, c-1]，而我们希望数据为[1, c]，所以要加一
                index = c * j + c - hash_value1   # 一个one-hot编码中唯一的1对应的标号：例如对j=1（对应cws的第一个sample:i1），对应one-hot编码中为1的位置：n - i1
                X[i, index] = 1
            for t in range(samples1.shape[1], k):
                sample_value2 = samples2[i, t - samples1.shape[1]]
                if (sample_value2 == -1 or sample_value2 == 0):
                    sum_zero_sample2 += 1
                    continue
                assert sample_value2 >= 1
                hash_value2 = (a_lst[t] * (sample_value2 + n1) + b_lst[t]) % c + 1
                index = c * t + c - hash_value2
                X[i, index] = 1
        print('采样为0的情况出现的次数: ', sum_zero_sample1, sum_zero_sample2)
        print('全0instance: ', sum_zero_instance1, sum_zero_instance2)
        X = X.tocsr()
        return X
    # 3 进行b-bit minwise hash特征压缩
    elif (b != 0 and c == 0):
        X = sp.lil_matrix((m, 2 ** b * k), dtype=int)
        sum_zero_sample1 = 0  # 统计sample=0出现的次数
        sum_zero_sample2 = 0  # 统计sample=0出现的次数
        sum_zero_instance1 = 0  # 统计全0instance
        sum_zero_instance2 = 0  # 统计全0instance
        for i in range(m):
            if (samples1[i, 0] == -1 or samples1[i, 0] == 0):
                sum_zero_instance1 += 1
            if (samples2[i, 0] == -1 or samples2[i, 0] == 0):
                sum_zero_instance2 += 1
            for j in range(samples1.shape[1]):
                sample_value1 = samples1[i, j]
                if (sample_value1 == -1 or sample_value1 == 0):
                    sum_zero_sample1 += 1
                    continue
                assert sample_value1 >= 1
                minwise_value1 = sample_value1 % (
                            2 ** b) + 1  # minwise hash：经过minwise hash，将数据从[1,n]映射成了[0, 2**b-1]，而我们希望数据为[1, 2**b]，所以要加一
                index = 2 ** b * j + 2 ** b - minwise_value1  # 一个one-hot编码中唯一的1对应的标号：例如对j=1（对应cws的第一个sample:i1），对应one-hot编码中为1的位置：n - i1
                X[i, index] = 1
            for t in range(samples1.shape[1], k):
                sample_value2 = samples2[i, t - samples1.shape[1]]
                if (sample_value2 == -1 or sample_value2 == 0):
                    sum_zero_sample2 += 1
                    continue
                assert sample_value2 >= 1
                minwise_value2 = (sample_value2 + n1) % (2 ** b) + 1
                index = 2 ** b * t + 2 ** b - minwise_value2
                X[i, index] = 1
        print('采样为0的情况出现的次数: ', sum_zero_sample1, sum_zero_sample2)
        print('全0instance: ', sum_zero_instance1, sum_zero_instance2)
        X = X.tocsr()
        return X

    else:
        print('参数b,c错误!')
        return 'ERROR'



PATH_DATA = '../data/'

def sketch_to_countsketch(tag, b, c, dataset, portion_method, portion, sampling_rate, sketching_method):
    """
    dataset结构:
    dataset_name/portion比例_近似方法/采样次数/4个数据集文件
    """
    sketch_name = "sketch" + str(sampling_rate) # sketch1024 or sketch512
    dataset_file_name = os.path.join(dataset, portion_method, sketch_name)
    # example: dataset_file_name = "DailySports/portion37_pminhash/sketch1024/""

    # if sampling_rate == 1024:
    #     dataset_file_name = 'DailySports/portion37_pminhash/sketch1024/'
    # elif sampling_rate == 512:
    #     dataset_file_name = 'DailySports/portion37_pminhash/sketch512/'
    
    train_file_name1 = 'X1_train_samples.txt'
    train_file_name2 = 'X2_train_samples.txt'
    test_file_name1 = 'X1_test_samples.txt'
    test_file_name2 = 'X2_test_samples.txt'
    # dataset_file_name = 'splice/distrubuted/'  
    # train_file_name1 = 'X1_train_sketch.txt'
    # train_file_name2 = 'X2_train_sketch.txt'
    # test_file_name1 = 'X1_test_sketch.txt'
    # test_file_name2 = 'X2_test_sketch.txt'
    
    # dataset_file_name = 'splice/distrubuted/encoded'  
    # train_file_name1 = 'X1_encoded_train37.txt'
    # train_file_name2 = 'X2_encoded_train37.txt'
    # test_file_name1 = 'X1_encoded_test37.txt'
    # test_file_name2 = 'X2_encoded_test37.txt'
    # main_path = '/Users/zbz/code/vscodemac_python/hetero_sshe_logistic_regression/data/'
    main_path = PATH_DATA

    if portion == "37": partition = 3/10
    elif portion == "28": partition = 2/10
    elif portion == "19": partition = 1/10
    elif portion == "46": partition = 4/10
    elif portion == "55": partition = 5/10
    else: raise ValueError
    
    if tag == "train":
        X_train1 = np.loadtxt(os.path.join(main_path, dataset_file_name, train_file_name1), delimiter=',', dtype = int)
        X_train2 = np.loadtxt(os.path.join(main_path, dataset_file_name, train_file_name2), delimiter=',', dtype = int)
        # X_train = np.loadtxt('../data/splice/X_train_sketch.txt', delimiter=',') #, dtype = float)
        m = X_train1.shape[0]
        n = X_train1.shape[1] + X_train2.shape[1]
        print("Origin train data (samples) shape: ", m, n)
        n1 = X_train1.shape[1]
        k = n
        b = 0
        print("k = ", k)
        print("c = ", c)
        # import sys
        # sys.exit()

        sketch = Dis_samples_to_sketch(m, n, n1, k, b, c, X_train1, X_train2).toarray()
        print("Sketch train data shape: ", sketch.shape)

        k = sketch.shape[1]
        # partition = 3/10
        k1 = np.floor(k * partition).astype(int)
        result_X_train1, result_X_train2 = sketch[:,0:k1], sketch[:,k1:]

        # 训练集
        train_sketch_savepath = str(sketching_method) + "_" + str(c) # "countsketch"
        train_save_name1 = "X1_squeeze_train37_Countsketch.txt"
        train_save_name2 = "X2_squeeze_train37_Countsketch.txt"
        
        np.savetxt(os.path.join(main_path, dataset_file_name, train_sketch_savepath, train_save_name1),
                    result_X_train1, delimiter=',', fmt='%d')
        np.savetxt(os.path.join(main_path, dataset_file_name, train_sketch_savepath, train_save_name2),
                    result_X_train2, delimiter=',', fmt='%d')
        

        # if dataset == "DailySports":
        #     if sampling_rate == 1024:
        #         np.savetxt("../data/DailySports/portion37_pminhash/sketch1024/countsketch/X1_squeeze_train37_Countsketch.txt",
        #                     result_X_train1, delimiter=',', fmt='%d')
        #         np.savetxt("../data/DailySports/portion37_pminhash/sketch1024/countsketch/X2_squeeze_train37_Countsketch.txt",
        #                     result_X_train2, delimiter=',', fmt='%d')
        #     elif sampling_rate == 512:
        #         np.savetxt("../data/DailySports/portion37_pminhash/sketch512/countsketch/X1_squeeze_train37_Countsketch.txt",
        #                     result_X_train1, delimiter=',', fmt='%d')
        #         np.savetxt("../data/DailySports/portion37_pminhash/sketch512/countsketch/X2_squeeze_train37_Countsketch.txt",
        #                     result_X_train2, delimiter=',', fmt='%d')
        #     else:
        #         raise NotImplementedError("Invalid sampling_rate.")

        print("Train count-sketch data saved.")

    elif tag == "test":
        X_test1 = np.loadtxt(os.path.join(main_path, dataset_file_name, test_file_name1), delimiter=',', dtype = int)
        X_test2 = np.loadtxt(os.path.join(main_path, dataset_file_name, test_file_name2), delimiter=',', dtype = int)
        # X_test = np.loadtxt('../data/splice/X_test_sketch.txt', delimiter=',')
        m = X_test1.shape[0]
        n = X_test1.shape[1] + X_test2.shape[1]
        print("Origin test data (samples) shape: ", m, n)
        n1 = X_test1.shape[1]
        k = n
        b = 0
        print("k = ", k)
        print("c = ", c)

        sketch = Dis_samples_to_sketch(m, n, n1, k, b, c, X_test1, X_test2).toarray()
        print("Sketch test data shape: ", sketch.shape)

        k = sketch.shape[1]
        # partition = 3/10
        k1 = np.floor(k * partition).astype(int)
        result_X_test1, result_X_test2 = sketch[:,0:k1], sketch[:,k1:]

        print(result_X_test1.shape[0], result_X_test1.shape[1])
        print(result_X_test2.shape[0], result_X_test2.shape[1])

        # 测试集
        test_sketch_savepath = str(sketching_method) + "_" + str(c)  # "countsketch"
        test_save_name1 = "X1_squeeze_test37_Countsketch.txt"
        test_save_name2 = "X2_squeeze_test37_Countsketch.txt"
        
        np.savetxt(os.path.join(main_path, dataset_file_name, test_sketch_savepath, test_save_name1),
                    result_X_test1, delimiter=',', fmt='%d')
        np.savetxt(os.path.join(main_path, dataset_file_name, test_sketch_savepath, test_save_name2),
                    result_X_test2, delimiter=',', fmt='%d')
        
        # if dataset == "DailySports":
        #     if sampling_rate == 1024:
        #         np.savetxt("../data/DailySports/portion37_pminhash/sketch1024/countsketch/X1_squeeze_test37_Countsketch.txt",
        #                     result_X_test1, delimiter=',', fmt='%d')
        #         np.savetxt("../data/DailySports/portion37_pminhash/sketch1024/countsketch/X2_squeeze_test37_Countsketch.txt",
        #                     result_X_test2, delimiter=',', fmt='%d')
        #     elif sampling_rate == 512:
        #         np.savetxt("../data/DailySports/portion37_pminhash/sketch512/countsketch/X1_squeeze_test37_Countsketch.txt",
        #                     result_X_test1, delimiter=',', fmt='%d')
        #         np.savetxt("../data/DailySports/portion37_pminhash/sketch512/countsketch/X2_squeeze_test37_Countsketch.txt",
        #                     result_X_test2, delimiter=',', fmt='%d')
        #     else:
        #         raise NotImplementedError("Invalid sampling_rate.")

        print("Test count-sketch data saved.")
    else:
        raise Exception('[Exception] tag error occurred.')
    



if __name__ == '__main__':
    # X_train1, X_train2, Y_train, X_test1, X_test2, Y_test = read_distributed_encoded_data()
    print("loading dataset...")
    
    dataset_name = "SVHN" # DailySports / kits / robert / cifar10 / SVHN
    # sampling_rate = 512
    # c = 4
    portion = "37"
    portion_method = "portion37_pminhash"
    sketching_method = "countsketch"
    # sketch_to_countsketch("train", 0, c, dataset_name, portion_method, portion, sampling_rate, sketching_method)
    # sketch_to_countsketch("test", 0, c, dataset_name, portion_method, portion, sampling_rate, sketching_method)
    
    for sampling_rate in [512, 1024]:
        for c in [2, 4]:
            sketch_to_countsketch("train", 0, c, dataset_name, portion_method, portion, sampling_rate, sketching_method)
            sketch_to_countsketch("test", 0, c, dataset_name, portion_method, portion, sampling_rate, sketching_method)
    