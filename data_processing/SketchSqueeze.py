import numpy as np
import scipy.sparse as sp

# 集中式
def Con_samples_to_sketch(m, n, k, b, c, samples):
    """
    :param m: 样本数
    :param n: 特征数
    :param k: P-minhash和0 bit CWS的采样次数（采样结果samples1，samples2的列数之和）
    :param b: b bit minwise hash的b（将特征数n压缩至2^b）
    :param c: Count Sketch的c（将特征数n压缩至c）
    :param samples: 数据集的CWS采样结果
    :return: 将P-minhash和0 bit CWS得到的sample转为sketch
    """
    assert m == samples.shape[0]  # 若samples的样本数与原数据样本数不一致，触发异常
    assert k == samples.shape[1]  # 采样次数k与samples的列数不一致，则触发异常
    # 1 不进行任何特征压缩
    if (b == 0 and c == 0):
        X = sp.lil_matrix((m, n * k), dtype=int)  # lil_matrix（适合用于替换值）数据为int型的稀疏矩阵
        sum_zero_sample = 0                        # 统计全0特征的instance出现的次数
        for i in range(m):                         # 遍历(m,k)的sketch二维数组
            for j in range(k):
                sample_value = samples[i, j]       # CWS采样结果值（int，正常范围为[1,n]，若原数据某instance为特征全0，则采样结果为-1或0，我们定义此时的one-hot编码为全0
                if (sample_value == -1 or sample_value == 0):  # 统计全0特征的instance出现的次数
                    sum_zero_sample += 1
                    continue
                assert sample_value >= 1
                index = n * j + n - sample_value   # 一个one-hot编码中唯一的1对应的标号：例如对j=1（对应cws的第一个sample:i1），对应one-hot编码中为1的位置：n - i1
                X[i, index] = 1
        print('采样为0的情况出现的次数：', sum_zero_sample)
        X = X.tocsr()
        return X

    # 2 进行Count Sketch特征压缩
    elif (b == 0 and c != 0):
        X = sp.lil_matrix((m, c * k), dtype=int)
        sum_zero_sample = 0
        # 创建随机数组用于构成哈希函数h1~hk
        np.random.seed(1)
        a_lst = np.random.randint(low=1, high=c, size=k)
        b_lst = np.random.randint(low=0, high=c, size=k)
        for i in range(m):
            for j in range(k):
                sample_value = samples[i, j]
                if (sample_value == -1 or sample_value == 0):
                    sum_zero_sample += 1
                    continue
                assert sample_value >= 1
                hash_value = (a_lst[j] * sample_value + b_lst[j]) % c + 1  # 经过minwise hash，将数据从[1,n]映射成了[0, c-1]，而我们希望数据为[1, c]，所以要加一
                index = c * j + c - hash_value  # 一个one-hot编码中唯一的1对应的标号：例如对j=1（对应cws的第一个sample:i1），对应one-hot编码中为1的位置：n - i1
                X[i, index] = 1
        print('采样为0的情况出现的次数：', sum_zero_sample)
        X = X.tocsr()
        return X

    # 3 进行b-bit minwise hash特征压缩
    elif (b != 0 and c == 0):
        X = sp.lil_matrix((m, 2 ** b * k), dtype=int)
        sum_zero_sample = 0
        for i in range(m):
            for j in range(k):
                sample_value = samples[i, j]
                if (sample_value == -1 or sample_value == 0):
                    sum_zero_sample += 1
                    continue
                assert sample_value >= 1
                # b-bit minwise hash
                minwise_value = sample_value % (2 ** b) + 1  # 经过minwise hash，将数据从[1,n]映射成了[0, 2**b-1]，而我们希望数据为[1, 2**b]，所以要加一
                index = 2 ** b * j + 2 ** b - minwise_value  # 一个one-hot编码中唯一的1对应的标号：例如对j=1（对应cws的第一个sample:i1），对应one-hot编码中为1的位置：n - i1
                X[i, index] = 1
        print('采样为0的情况出现的次数：', sum_zero_sample)
        X = X.tocsr()
        return X
    else:
        print('参数b,c错误！')
        return 'ERROR'



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

        print('采样为0的情况出现的次数：', sum_zero_sample1, sum_zero_sample2)
        print('全0instance：', sum_zero_instance1, sum_zero_instance2)
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
        print('采样为0的情况出现的次数：', sum_zero_sample1, sum_zero_sample2)
        print('全0instance：', sum_zero_instance1, sum_zero_instance2)
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
        print('采样为0的情况出现的次数：', sum_zero_sample1, sum_zero_sample2)
        print('全0instance：', sum_zero_instance1, sum_zero_instance2)
        X = X.tocsr()
        return X

    else:
        print('参数b,c错误！')
        return 'ERROR'