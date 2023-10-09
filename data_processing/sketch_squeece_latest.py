import numpy as np
import scipy.sparse as sp
import os
from sklearn.datasets import load_svmlight_file
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





# 互不牵涉的分布式sample to sketch
def Dis_samples_to_sketch2(m, n1, n2, k1, k2, b, c, samples1, samples2):
    """
    :param m: 样本数
    :param n: 特征数
    :param n1: 特征划分的第一部分数据集的最大特征数
    :param k: P-minhash和0 bit CWS的采样次数(采样结果samples1,samples2的列数之和)
    :param b: b bit minwise hash的b(将特征数n压缩至2^b)
    :param c: Count Sketch的c(将特征数n压缩至c)
    :param samples1: 特征划分的第一部分数据集的CWS采样结果
    :param samples2: 特征划分的第二部分数据集的CWS采样结果
    :return: 将P-minhash和0 bit CWS得到的sample转为sketch
    """

    assert m == samples1.shape[0]     # 若samples的样本数与原数据样本数不一致，触发异常
    assert m == samples2.shape[0]
    assert k1 == samples1.shape[1]
    assert k2 == samples2.shape[1]    # 采样次数k与samples的列数不一致，则触发异常
    # 1 不进行任何特征压缩
    if (b == 0 and c == 0):
        X1 = sp.lil_matrix((m, n1 * k1), dtype=int)  # lil_matrix（适合用于替换值）数据为int型的稀疏矩阵，行列数与原数据集的稀疏矩阵保持一致
        X2 = sp.lil_matrix((m, n2 * k2), dtype=int)
        sum_zero_sample1 = 0  # 统计sample=0出现的次数
        sum_zero_instance1 = 0  # 统计全0instance
        sum_zero_sample2 = 0  # 统计sample=0出现的次数
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
                index = n1 * j + n1 - sample_value1  # 一个one-hot编码中唯一的1对应的标号：例如对j=1（对应cws的第一个sample:i1），对应one-hot编码中为1的位置：n - i1
                X1[i, index] = 1
            for t in range(samples2.shape[1]):
                sample_value2 = samples2[i, t]
                if (sample_value2 == -1 or sample_value2 == 0):
                    sum_zero_sample2 += 1
                    continue
                assert sample_value2 >= 1
                index = n2 * t + n2 - sample_value2  # 一个one-hot编码中唯一的1对应的标号：例如对j=1（对应cws的第一个sample:i1），对应one-hot编码中为1的位置：n - i1
                X2[i, index] = 1

        X1 = X1.tocsr()
        X2 = X2.tocsr()
        X = sp.hstack([X1, X2])
        # print('采样为0的情况出现的次数：', sum_zero_sample1, sum_zero_sample2)
        # print('全0instance：', sum_zero_instance1, sum_zero_instance2)
        return X
    # 2 进行Count Sketch特征压缩
    elif (b == 0 and c != 0):
        if n1 <= c:
            X1 = sp.lil_matrix((m, n1 * k1), dtype=int)  # lil_matrix（适合用于替换值）数据为int型的稀疏矩阵，行列数与原数据集的稀疏矩阵保持一致
        else:
            X1 = sp.lil_matrix((m, c * k1), dtype=int)
        if n2 <= c:
            X2 = sp.lil_matrix((m, n2 * k2), dtype=int)
        else:
            X2 = sp.lil_matrix((m, c * k2), dtype=int)
        sum_zero_sample1 = 0  # 统计sample=0出现的次数
        sum_zero_sample2 = 0  # 统计sample=0出现的次数
        sum_zero_instance1 = 0  # 统计全0instance
        sum_zero_instance2 = 0  # 统计全0instance
        # 创建随机数组用于构成哈希函数h1~hk
        np.random.seed(1)
        a_lst = np.random.randint(low=1, high=c, size=k1+k2)
        b_lst = np.random.randint(low=0, high=c, size=k1+k2)
        for i in range(m):
            if (samples1[i, 0] == -1 or samples1[i, 0] == 0):
                sum_zero_instance1 += 1
            if (samples2[i, 0] == -1 or samples2[i, 0] == 0):
                sum_zero_instance2 += 1
            for j in range(k1):
                sample_value1 = samples1[i, j]
                if (sample_value1 == -1 or sample_value1 == 0):
                    sum_zero_sample1 += 1
                    continue
                assert sample_value1 >= 1
                if n1 <= c:
                    index = n1 * j + n1 - sample_value1
                else:
                    hash_value1 = (a_lst[j] * sample_value1 + b_lst[j]) % c + 1  # 经过Count Sketch哈希，将数据从[1,n]映射成了[0, c-1]，而我们希望数据为[1, c]，所以要加一
                    index = c * j + c - hash_value1  # 一个one-hot编码中唯一的1对应的标号：例如对j=1（对应cws的第一个sample:i1），对应one-hot编码中为1的位置：n - i1
                X1[i, index] = 1
            for t in range(k2):
                sample_value2 = samples2[i, t]
                if (sample_value2 == -1 or sample_value2 == 0):
                    sum_zero_sample2 += 1
                    continue
                assert sample_value2 >= 1
                if n2 <= c:
                    index = n2 * t + n2 - sample_value2
                else:
                    hash_value2 = (a_lst[k1+t] * sample_value2 + b_lst[k1+t]) % c + 1
                    index = c * t + c - hash_value2
                X2[i, index] = 1

        X1 = X1.tocsr()
        X2 = X2.tocsr()
        X = sp.hstack([X1, X2])
        # print('采样为0的情况出现的次数：', sum_zero_sample1, sum_zero_sample2)
        # print('全0instance：', sum_zero_instance1, sum_zero_instance2)
        return X

    # 3 进行b-bit minwise hash特征压缩
    elif (b != 0 and c == 0):
        if n1 <= c:
            X1 = sp.lil_matrix((m, n1 * k1), dtype=int)  # lil_matrix（适合用于替换值）数据为int型的稀疏矩阵，行列数与原数据集的稀疏矩阵保持一致
        else:
            X1 = sp.lil_matrix((m, 2 ** b * k1), dtype=int)
        if n2 <= c:
            X2 = sp.lil_matrix((m, n2 * k2), dtype=int)
        else:
            X2 = sp.lil_matrix((m, 2 ** b * k2), dtype=int)
        sum_zero_sample1 = 0  # 统计sample=0出现的次数
        sum_zero_sample2 = 0  # 统计sample=0出现的次数
        sum_zero_instance1 = 0  # 统计全0instance
        sum_zero_instance2 = 0  # 统计全0instance
        for i in range(m):
            if (samples1[i, 0] == -1 or samples1[i, 0] == 0):
                sum_zero_instance1 += 1
            if (samples2[i, 0] == -1 or samples2[i, 0] == 0):
                sum_zero_instance2 += 1
            for j in range(k1):
                sample_value1 = samples1[i, j]
                if (sample_value1 == -1 or sample_value1 == 0):
                    sum_zero_sample1 += 1
                    continue
                assert sample_value1 >= 1
                minwise_value1 = sample_value1 % (
                            2 ** b) + 1  # minwise hash：经过minwise hash，将数据从[1,n]映射成了[0, 2**b-1]，而我们希望数据为[1, 2**b]，所以要加一
                index = 2 ** b * j + 2 ** b - minwise_value1  # 一个one-hot编码中唯一的1对应的标号：例如对j=1（对应cws的第一个sample:i1），对应one-hot编码中为1的位置：n - i1
                X1[i, index] = 1
            for t in range(k2):
                sample_value2 = samples2[i, t]
                if (sample_value2 == -1 or sample_value2 == 0):
                    sum_zero_sample2 += 1
                    continue
                assert sample_value2 >= 1
                minwise_value2 = sample_value2 % (2 ** b) + 1
                index = 2 ** b * t + 2 ** b - minwise_value2
                X2[i, index] = 1
        X1 = X1.tocsr()
        X2 = X2.tocsr()
        X = sp.hstack([X1, X2])
        # print('采样为0的情况出现的次数：', sum_zero_sample1, sum_zero_sample2)
        # print('全0instance：', sum_zero_instance1, sum_zero_instance2)
        return X

    else:
        print('参数b,c错误!')
        return 'ERROR'












PATH_DATA = '/home/user/zbz/SSHE/data/'

def sample_to_countsketch(tag, b, c, dataset, kernel_approx, portion, sampling_rate, sketching_method):
    """
    dataset结构:
    dataset_name/portion比例_近似方法/采样次数/4个数据集文件
    读取采样结果samples, 直接转化为sketch(有countsketch/0bitminwisehash压缩 or 无压缩)
    """
    if tag == "train": print("================= Train =================")
    else: print("================= Test =================")
    print("sketch-{}, k={}; method-{}, b={}, c={}".format(kernel_approx, sampling_rate, sketching_method, b, c))

    sketch_name = "sketch" + str(sampling_rate) # sketch1024 or sketch512
    portion_method = "portion37" + "_" + str(kernel_approx) # portion37_pminhash / portion37_0bitcws
    dataset_file_name = os.path.join(dataset, portion_method, sketch_name)
    # example: dataset_file_name = "DailySports/portion37_pminhash/sketch1024/""
    
    train_file_name1 = 'X1_train_samples.txt'
    train_file_name2 = 'X2_train_samples.txt'
    test_file_name1 = 'X1_test_samples.txt'
    test_file_name2 = 'X2_test_samples.txt'

    # main_path = '/Users/zbz/code/vscodemac_python/hetero_sshe_logistic_regression/data/'
    main_path = PATH_DATA

    if portion == "37": partition = 3/10
    elif portion == "28": partition = 2/10
    elif portion == "19": partition = 1/10
    elif portion == "46": partition = 4/10
    elif portion == "55": partition = 5/10
    else: raise ValueError

    # kernel_approx = "pminhash"  # pminhash / 0bitcws / rff / poly
    if kernel_approx == "pminhash": # countsketch
        assert(b == 0)
        assert(sketching_method == "countsketch")
    elif kernel_approx == "0bitcws": # b-bit minwise hash
        assert(c == 0)
        assert(sketching_method == "bbitmwhash")


    """ 读取原数据集, 获取特征维度 """
    main_path = PATH_DATA
    dataset_Rawfile_name = dataset  
    train_Rawfile_name = dataset + '_train.txt' 
    test_Rawfile_name = dataset + '_test.txt'
    # dataset_file_name = 'DailySports'  
    # train_file_name = 'DailySports_train.txt' 
    # test_file_name = 'DailySports_test.txt'
    
    """ 训练集 """
    if tag == "train":
        X_train_samples_1 = np.loadtxt(os.path.join(main_path, dataset_file_name, train_file_name1), delimiter=',', dtype = int)
        X_train_samples_2 = np.loadtxt(os.path.join(main_path, dataset_file_name, train_file_name2), delimiter=',', dtype = int)
        # load_svmlight_file 输出稀疏矩阵格式
        train_data = load_svmlight_file(os.path.join(main_path, dataset_Rawfile_name, train_Rawfile_name))
        # X_train_Raw = train_data[0].todense().A

        m = X_train_samples_1.shape[0] # 样本数
        # print("yangebnshuu:;; ", m)
        k = X_train_samples_1.shape[1] + X_train_samples_2.shape[1] # 总采样次数
        k1 = X_train_samples_1.shape[1] # 第一部分采样次数
        k2 = X_train_samples_2.shape[1] # 第二部分采样次数
        """
        读取原数据集的特征维度
        """
        # n = X_train_Raw.shape[1]
        n = train_data[0].shape[1]
        n1 = np.floor(n * partition).astype(int)    # 这部分内容需要和sketch那部分的东西对应, 不过其实影响不太大?
        n2 = n - n1
        print("Origin train data (samples) shape: ", m, n)
        print("k = {}, c = {}, b = {}".format(k, c, b))
        
        """ Method 2 for countsketch (Pminhash, 0bitcws) """
        sketch = Dis_samples_to_sketch2(m, n1, n2, k1, k2, b, c, X_train_samples_1, X_train_samples_2).toarray()
        print("Sketch train data shape: ", sketch.shape)


        """ 数据集按照比例划分 """
        sk = sketch.shape[1]
        # partition = 3/10
        sk1 = np.floor(sk * partition).astype(int)
        result_X_train1, result_X_train2 = sketch[:,0:sk1], sketch[:,sk1:]


        if sketching_method == "countsketch":
            assert(sketch.shape[0] == m)
            assert(sketch.shape[1] == (c * k))
            # 训练集
            train_sketch_savepath = str(sketching_method) + "_" + str(c) # "countsketch"
            train_save_name1 = "X1_squeeze_train37.txt"
            train_save_name2 = "X2_squeeze_train37.txt"
            """
            # PATH eg.:  ../data/DailySports/portion37_pminhash/sketch1024/countsketch_2/
            """
            np.savetxt(os.path.join(main_path, dataset_file_name, train_sketch_savepath, train_save_name1),
                        result_X_train1, delimiter=',', fmt='%d')
            np.savetxt(os.path.join(main_path, dataset_file_name, train_sketch_savepath, train_save_name2),
                        result_X_train2, delimiter=',', fmt='%d')
            

        elif sketching_method == "bbitmwhash":
            assert(sketch.shape[0] == m)
            assert(sketch.shape[1] == ((2 ** b) * k))
            train_sketch_savepath = str(sketching_method) + "_" + str(2 ** b) # "bbitmwhash" b = [1,2,3]; 2**b=[2^1 2^2 2^3]=[2 4 8]
            train_save_name1 = "X1_squeeze_train37.txt"
            train_save_name2 = "X2_squeeze_train37.txt"

            """
            # PATH eg.:  ../data/ DailySports/portion37_0bitcws/sketch1024/ bbitmwhash_2/   # 这里的2=2**b=2**1=2^1,即b=1
            """
            np.savetxt(os.path.join(main_path, dataset_file_name, train_sketch_savepath, train_save_name1),
                        result_X_train1, delimiter=',', fmt='%d')
            np.savetxt(os.path.join(main_path, dataset_file_name, train_sketch_savepath, train_save_name2),
                        result_X_train2, delimiter=',', fmt='%d')

        print("Train count-sketch data saved.")


        
    elif tag == "test":
        """ 测试集 """
        X_test_samples_1 = np.loadtxt(os.path.join(main_path, dataset_file_name, test_file_name1), delimiter=',', dtype = int)
        X_test_samples_2 = np.loadtxt(os.path.join(main_path, dataset_file_name, test_file_name2), delimiter=',', dtype = int)
        test_data = load_svmlight_file(os.path.join(main_path, dataset_Rawfile_name, test_Rawfile_name))
        # X_test_Raw = test_data[0].todense().A

        m = X_test_samples_1.shape[0] # 样本数
        k = X_test_samples_1.shape[1] + X_test_samples_2.shape[1] # 总采样次数
        k1 = X_test_samples_1.shape[1] # 第一部分采样次数
        k2 = X_test_samples_2.shape[1] # 第二部分采样次数

        """
        读取原数据集的特征维度
        """
        # n = X_test_Raw.shape[1]
        n = test_data[0].shape[1]
        n1 = np.floor(n * partition).astype(int)    # 这部分内容需要和sketch那部分的东西对应, 不过其实影响不太大?
        n2 = n - n1
        print("Origin test data (samples) shape: ", m, n)
        print("k = {}, c = {}, b = {}".format(k, c, b))

        sketch = Dis_samples_to_sketch2(m, n1, n2, k1, k2, b, c, X_test_samples_1, X_test_samples_2).toarray()
        print("Sketch test data shape: ", sketch.shape)



        """ 数据集按照比例划分 """
        sk = sketch.shape[1]
        # partition = 3/10
        sk1 = np.floor(sk * partition).astype(int)
        result_X_test1, result_X_test2 = sketch[:,0:sk1], sketch[:,sk1:]

        print(result_X_test1.shape[0], result_X_test1.shape[1])
        print(result_X_test2.shape[0], result_X_test2.shape[1])

        if sketching_method == "countsketch":
            assert(sketch.shape[0] == m)
            assert(sketch.shape[1] == (c * k))
            # 测试集
            test_sketch_savepath = str(sketching_method) + "_" + str(c)  # "countsketch"
            test_save_name1 = "X1_squeeze_test37.txt"
            test_save_name2 = "X2_squeeze_test37.txt"
            """
            # PATH eg.:  ../data/DailySports/portion37_pminhash/sketch1024/countsketch_2/
            """
            np.savetxt(os.path.join(main_path, dataset_file_name, test_sketch_savepath, test_save_name1),
                        result_X_test1, delimiter=',', fmt='%d')
            np.savetxt(os.path.join(main_path, dataset_file_name, test_sketch_savepath, test_save_name2),
                        result_X_test2, delimiter=',', fmt='%d')
            

        elif sketching_method == "bbitmwhash":
            assert(sketch.shape[0] == m)
            assert(sketch.shape[1] == ((2 ** b) * k))
            # 测试集
            test_sketch_savepath = str(sketching_method) + "_" + str(2 ** b)  # "bbitmwhash" b = [1,2,3]; 2**b=[2^1 2^2 2^3]=[2 4 8]
            test_save_name1 = "X1_squeeze_test37.txt"
            test_save_name2 = "X2_squeeze_test37.txt"
            """
            # PATH eg.:  ../data/DailySports/portion37_0bitcws/sketch1024/bbitmwhash_2/   # 这里的2=2**b=2**1=2^1,即b=1
            """
            np.savetxt(os.path.join(main_path, dataset_file_name, test_sketch_savepath, test_save_name1),
                        result_X_test1, delimiter=',', fmt='%d')
            np.savetxt(os.path.join(main_path, dataset_file_name, test_sketch_savepath, test_save_name2),
                        result_X_test2, delimiter=',', fmt='%d')


        print("Test count-sketch data saved.")
        print("========================================")
    else:
        raise Exception('[Exception] tag error occurred.')
    



if __name__ == '__main__':
    # X_train1, X_train2, Y_train, X_test1, X_test2, Y_test = read_distributed_encoded_data()
    print("loading dataset...")
    
    dataset_name = "webspam10k" # DailySports (ok) / kits / robert ok / cifar10 ok / SVHN (ok)
    print("dataset: ", dataset_name)
    # sampling_rate = 512
    # c = 4
    portion = "37"
    # portion_method = "portion37_pminhash"
    # kernel_approx = "pminhash"  # pminhash / 0bitcws / rff / poly
    # sketching_method = "countsketch"  # countsketch / bbitmwhash
    # sample_to_countsketch("train", 0, c, dataset_name, portion_method, kernel_approx, portion, sampling_rate, sketching_method)
    # sample_to_countsketch("test", 0, c, dataset_name, portion_method, kernel_approx, portion, sampling_rate, sketching_method)
    
    # webspam c=8,16, b = 3/4 k = 2048,4096

    # for kernel_approx in ["pminhash", "0bitcws"]:
    for kernel_approx in ["0bitcws"]:
        # for sampling_rate in [512, 1024]:
        for sampling_rate in [2048, 4096]:

            if kernel_approx == "pminhash": 
                sketching_method = "countsketch"
                b = 0
                # for c in [8, 16]: 
                for c in [2, 4]:
                    sample_to_countsketch("train", b, c, dataset_name, kernel_approx, portion, sampling_rate, sketching_method)
                    sample_to_countsketch("test", b, c, dataset_name, kernel_approx, portion, sampling_rate, sketching_method)

            elif kernel_approx == "0bitcws": 
                sketching_method = "bbitmwhash"
                c = 0
                # for b in [3, 4]: 
                for b in [1, 2]:
                    sample_to_countsketch("train", b, c, dataset_name, kernel_approx, portion, sampling_rate, sketching_method)
                    sample_to_countsketch("test", b, c, dataset_name, kernel_approx, portion, sampling_rate, sketching_method)


"""
现在的数据集文件目录格式说明:

portion37_pminhash, portion37_0bitcws 指这里的数据sketch是按照纵向3:7的比例进行特征划分, 使用的采样方法为 pminhash/0bitcws

sketch1024,sketch512 指这里的数据sketch是原数据集采样1024/512次得到的.

countsketch_2 / _4 的2和4代表的是countsketch的参数c=2/4, 且最终c*k就是sketch数据集列数
bbitmwhash_2 / _4 的2和4代表的是bbitminwisehash的参数b=1/2, 2^b=2/4, 最终(2^b)*k就是sketch数据集列数

"""