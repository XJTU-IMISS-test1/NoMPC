"""
SecureML OVR 版本, 尝试将数据处理部分放在循环外部
后期将 数据的process+loader \ 模型定义 \ 训练 互相解耦
目前的依赖模块: 
E:\zbz\code\vscode_python\hetero_sshe_logistic_regression\paillierm
E:\zbz\code\vscode_python\hetero_sshe_logistic_regression\data_processing\dataset_select.py
E:\zbz\code\vscode_python\hetero_sshe_logistic_regression\data_processing\kernel_approximate_method.py
"""

import numpy as np
import time
import argparse
# import matplotlib.pyplot as plt
# from scipy.sparse import csr_matrix, lil_matrix, coo_matrix
from multiprocessing import Pool, Process, Queue
# import multiprocessing

import sys
import os
dir_path = os.path.dirname(os.path.realpath(__file__))
abs_pardir = os.path.join(dir_path, os.pardir)
abs_parpardir = os.path.join(abs_pardir, os.pardir)
sys.path.append(abs_parpardir)
# print(abs_parpardir)
from paillierm.encrypt import PaillierEncrypt
from paillierm.fixedpoint import FixedPointEndec
from paillierm.utils import urand_tensor

""" data processing and loading method """
from data_processing.dataset_select import dataset_selector
# from data_processing.sketch_squeece import *
from data_processing.kernel_approximate_method import dataSketch_generator_RBFandPolyKernel


PATH_DATA = '../../data/' # '../../data/'


class SecureML:
    """
    SecureML Implementation
    """
    def __init__(self, weight_vector, batch_size, max_iter, alpha, 
                        eps, ratio = None, penalty = None, lambda_para = 1, data_tag = None, ovr = None,
                        sketch_tag = None, countsketch_c = 0, bbitmwhash_b = 0, dataset_name = None, 
                        kernel_method = None, sampling_k = None, Epoch_list_max = None):
        """
        构造函数:初始化
        """
        self.model_weights = weight_vector
        self.batch_size = batch_size # 设置的batch大小
        self.batch_num = [] # 存储每个batch的大小
        self.n_iteration = 0
        self.max_iter = max_iter
        self.alpha = alpha
        self.pre_loss = 0
        self.eps = eps # 训练的误差下限
        self.ratio = ratio # 数据集划分比例
        self.penalty = penalty # 正则化策略
        self.lambda_para = lambda_para # 正则化系数
        self.data_tag = data_tag # 输入数据的格式 (目前支持两种格式: sparse和dense)
        self.ovr = ovr
        self.countsketch_c = countsketch_c  # countsketch的c值
        self.bbitmwhash_b = bbitmwhash_b # bbitminwisehash的b值, 数据集存储目录的后缀写的是2^b
        self.kernel_method = kernel_method
        
        # WAN(Wide area network) Bandwidth, unit: 使用单位: Mbps (1 MB/s = 8 Mbps); 带宽测试: 40Mbps (5MB/s)
        self.WAN_bandwidth = 10 # Mbps
        self.online_comm_time_account = 0 # online通信耗时
        self.offline_comm_time_account = 0  # offline通信耗时
        self.inference_time_account = 0 # 模型推理部分的时间统计
        self.inference_time_account_off = 0 # inference total
        self.mem_occupancy = 8 # B 字节 
        # 计算时: 元素个数 * 4 B / 1024 / 1024 MB  / (40/8) s = object_num * self.mem_occupancy / (1024*1024) / (self.WAN_bandwidth/8)

        ## compute time account
        self.offline_calculate_time = 0

        # 加密部分的初始化
        self.cipher = PaillierEncrypt() # Paillier初始化
        self.cipher.generate_key()  # Paillier生成公私钥
        self.fixedpoint_encoder = FixedPointEndec(n = 1e10) # 加密前的定点化编码器初始化

        # Epoch exp
        
        EPOCH_list = []
        # Epoch_list_max = 41
        # for epoch_num in range(0, Epoch_list_max): # 前包含后不包含, 后面取的数大于最终需要的数就行
        #     if epoch_num % 5 == 0:
        #         EPOCH_list.append(epoch_num)
        # EPOCH_list[0] = 1
        EPOCH_list = [i for i in range(1, Epoch_list_max)]

        self.EPOCH_list = EPOCH_list # eg. [1,5,10,15,20,25,30,35,40]
        assert(self.EPOCH_list[-1] <= self.max_iter)



        # Epoch recordings
        if self.ovr == "bin": self.modelWeight_and_Time_List = dict()
        elif self.ovr == "ovr": self.OVRModel_Agg = dict() # 用于存储每个标签对应的二分类模型, 其中每一个标签下的模型也是一个字典, 字典内容是不同epoch下的模型参数和时间

        import time
        filename = time.strftime("%Y-%m-%d-%H_%M_%S", time.localtime(time.time()))
        if sketch_tag == "sketch":
            if kernel_method == "pminhash":
                self.logname = "SecureML_" + dataset_name + "_" + kernel_method + sampling_k + "_" + str(countsketch_c) + "_" + filename + ".txt"
            elif kernel_method == "0bitcws":
                self.logname = "SecureML_" + dataset_name + "_" + kernel_method + sampling_k + "_" + str(2**bbitmwhash_b) + "_" + filename + ".txt"
            else:
                # rff, poly
                self.logname = "SecureML_" + dataset_name + "_" + kernel_method + sampling_k + "_" + filename + ".txt"
        else:
            self.logname = "SecureML_" + dataset_name + "_raw_" + filename + ".txt"

        # 进程池
        # self.pool = Pool()

    def _compute_sigmoid(self, z):
        # return 1 / (1 + np.exp(-z))
        # print(type(z))
        # if self.data_tag == None: 
        return z * 0.25 + 0.5
        # elif self.data_tag == 'sparse': return z.todense() * 0.25 + 0.5
    
    def _compute_sigmoid_dual_distributed(self, z):
        # return 1 / (1 + np.exp(-z))
        # print(type(z))
        # if self.data_tag == None: 
        return z * 0.25

    def check_converge_by_loss(self, loss):
        converge_flag = False
        if self.pre_loss is None:
            pass
        elif abs(self.pre_loss - loss) < self.eps:
            converge_flag = True
        self.pre_loss = loss
        return converge_flag
    
    def shuffle_distributed_data(self, XdatalistA, XdatalistB, y_batch_listA, y_batch_listB,
                                 E_batch_list, Z0_batch_list, Z1_batch_list, Z_p0_batch_list, Z_p1_batch_list):
        zip_list = list( zip(XdatalistA, XdatalistB, y_batch_listA, y_batch_listB, 
                             E_batch_list, Z0_batch_list, Z1_batch_list, Z_p0_batch_list, Z_p1_batch_list) )              # 将a,b整体作为一个zip,每个元素一一对应后打乱
        
        np.random.shuffle(zip_list)    # 打乱c

        XdatalistA[:], XdatalistB[:], y_batch_listA[:], y_batch_listB[:], E_batch_list[:], Z0_batch_list[:], Z1_batch_list[:], Z_p0_batch_list[:], Z_p1_batch_list[:] = zip(*zip_list)

        return XdatalistA, XdatalistB, y_batch_listA, y_batch_listB, E_batch_list, Z0_batch_list, Z1_batch_list, Z_p0_batch_list, Z_p1_batch_list

    
    def shuffle_lists(self, *lists):
        import random
        """
        Shuffles multiple lists in corresponding order and returns the shuffled lists.

        Args:
        *lists: Variable number of lists to shuffle.

        Returns:
        A tuple containing the shuffled lists in the same order as input.
        """

        # Get the length of the first list to ensure all lists have the same length
        n = len(lists[0])

        # Check that all lists have the same length
        for lst in lists:
            # print("list len: ", len(lst))
            if len(lst) != n:
                raise ValueError("All lists must have the same length.")

        # import sys
        # sys.exit()

        # Get a list of indices to shuffle in the same order for all lists
        indices = list(range(n))
        random.shuffle(indices)

        # Shuffle each list using the same indices
        shuffled_lists = []
        for lst in lists:
            shuffled_lists.append([lst[i] for i in indices])

        return tuple(shuffled_lists)




    def time_counting(self, tensor, flag = None):
        commTime = 0
        # 计算tensor在WAN下传输的时间
        if flag == "offline":
            if tensor.ndim == 2:
                object_num = tensor.shape[0] * tensor.shape[1]
            else:
                object_num = tensor.shape[0]
            commTime = object_num * self.mem_occupancy / (1024*1024) / (self.WAN_bandwidth/8)
            self.offline_comm_time_account += commTime
        
        else:
            """ 
            online comm time accounting, save to self.online_comm_time_account
            """
            if tensor.ndim == 2:
                object_num = tensor.shape[0] * tensor.shape[1]
            else:
                object_num = tensor.shape[0]
            commTime = object_num * self.mem_occupancy / (1024*1024) / (self.WAN_bandwidth/8)
            # print("commTime: ", commTime)
            self.online_comm_time_account += commTime

    def time_counting_model_inference(self, tensor, flag = None):
        commTime = 0
        # 计算tensor在WAN下传输的时间
        if flag == "offline":
            if tensor.ndim == 2:
                object_num = tensor.shape[0] * tensor.shape[1]
            else:
                object_num = tensor.shape[0]
            commTime = object_num * self.mem_occupancy / (1024*1024) / (self.WAN_bandwidth/8)
            self.inference_time_account += commTime

    def secret_share_vector_plaintext(self, share_target, flag = None):
        '''
        Desc: 秘密分享(输入的share_target是明文)
        '''
         # 生成本地share向量
        _pre = urand_tensor(q_field = self.fixedpoint_encoder.n, tensor = share_target)
        tmp = self.fixedpoint_encoder.decode(_pre)  # 对每一个元素decode再返回, shape不变
        share = share_target - tmp
        self.time_counting(share, flag)
        return tmp, share # 返回的第一个参数是留在本方的share, 第二个参数是需要分享的share


    def secure_distributed_compute_loss_cross_entropy(self, label, Y_predictA, Y_predictB, batch_num):
        """
        Input
        -----
        label, Y_predictA(wxa), Y_predictB(wxb), batch_num

        Desc
        -----
            Use Taylor series expand log loss:
            Loss = - y * log(h(x)) - (1-y) * log(1 - h(x)) where h(x) = 1/(1+exp(-wx))
            Then loss' = - (1/N)*∑ ( log(1/2) - 1/2*wx + ywx -1/8(wx)^2 )
        """
        # self.encrypted_wx = self.wx_self_A + self.wx_self_B
        wx = Y_predictA + Y_predictB
        # print("wx: ", wx)
        # print("wx shape: ", wx.shape)
        # print("label: ", label)
        # print("label shape: ", label.shape)

        # import sys
        # sys.exit()

        half_wx = -0.5 * wx
        assert(wx.shape[0] == label.shape[0])
        ywx = wx * label

        # wx_square = (self.za * self.za + 2 * self.za * self.zb + self.zb * self.zb) * -0.125
        wx_square = (Y_predictA * Y_predictA + 2 * Y_predictA * Y_predictB + Y_predictB * Y_predictB) * -0.125
        # wx_square = np.dot(self.wx_self.T, self.wx_self) * -0.125 # 这里后续要修改，两方平方，有交叉项
        # assert all(wx_square == wx_square2)  # 数组比较的返回值为: 类似[True False False]

        loss = np.sum( (half_wx + ywx + wx_square) * (-1 / batch_num) - np.log(0.5) )
        # loss = np.sum( (half_wx + ywx + wx_square) * (-1 / batch_num) )
        return loss
    


    def secretSharing_Data_and_Labels(self, data_matrixA, data_matrixB, Y_train):
        '''
        将数据X和标签Y, 分享到两方.
        '''
        # print("sharing data and label: ")
        local_dataA, share_dataA = self.secret_share_vector_plaintext(data_matrixA)
        local_dataB, share_dataB = self.secret_share_vector_plaintext(data_matrixB)
        local_Y, share_Y = self.secret_share_vector_plaintext(Y_train)
        

        self.local_matrix_A = np.hstack((local_dataA, share_dataB))
        self.local_matrix_B = np.hstack((share_dataA, local_dataB))
        self.Y_A = local_Y
        self.Y_B = share_Y
        assert(self.local_matrix_A.shape == self.local_matrix_B.shape)
        print("Sharing raw data: \033[32mOK\033[0m")

    

    def reconstruct(self, Ei, Ei_):
        E = Ei + Ei_ # 两方都各自重建E
        # print("reconstruct: * 2")
        self.time_counting(Ei)
        self.time_counting(Ei_)
        return E

    def generate_UVZV_Z_multTriplets_beaver_triplets(self, n, d, t, B):
        """
        Generate beaver_triplets and ss to two parties A and B. (Offline phase)

        Parameters
        ---------
        `X` - `n*d`input data matrix
        n: 样本数
        d: 总特征维度
        t: 每个epoch对应的batch数, t = n/B 上取整 np.seiling
        B: batch size 大小

        Return
        ---------
        `U0`, `U1`, `V0`, `V1`, `V0_`, `V1_`, `Z0`, `Z1`, `Z0_`, `Z1_`; _0 for Party 0 and _1 for Party 1.
        """
        flag = "offline"
        self.U = np.random.rand(n, d)
        V = np.random.rand(d, t)
        # self.V_ = np.random.rand(B, t)
        self.U0, self.U1 = self.secret_share_vector_plaintext(self.U, flag)
        self.V0, self.V1 = self.secret_share_vector_plaintext(V, flag)

        # self.V0_, self.V1_ = self.secret_share_vector_plaintext(self.V_, flag)


        self.Z = np.dot(self.U, V) # 按照下面两行写, 乘法缺项
        # self.Z0 = np.dot(self.U0, self.V0)
        # self.Z1 = np.dot(self.U1, self.V1)
        self.Z0, self.Z1 = self.secret_share_vector_plaintext(self.Z, flag)

        # 这里Z_必须在训练中生成, 因为Z_对应的是每个的batch, 而不是整个数据集, Z_的维度: (|B|, t), 但是实际中, 这部分可以offline的去做, 提前沟通ndt和B就行, U V Z V' Z'都是offline(论文原话)
        # 注意遇到某个列不足|B|的长度时, Z_的生成需要注意维度: 此时Z
        # self.Z0_ = np.dot(self.U0.transpose(), self.V0_)
        # self.Z1_ = np.dot(self.U0.transpose(), self.V1_)
        # self.Z_ = np.dot(self.U.transpose(), self.V_) # !!!!不对, 注意Z'矩阵的生成方式(矩阵乘法的维度),如果直接乘, 从原理上就不对, 必须一个batch一个batch乘

        # return U0, U1, V0, V1, Z0, Z1, V0_, V1_, Z0_, Z1_
        # 参考FATE或

    """
    TODO
    1 分享数据, 初始化模型参数.
    2 生成batch data, 且需要A和B的batch是对应的
    3 生成乘法三元组(简化版实现), U0 U1 V0 V1 Z0 Z1 V0' V1' Z0' Z1'

    E: n*d
    F: d*t

    TODO:目前修改到: generate batch 要加上E, 还有计算Y*预测值

    开始训练:
    Ei = Xi - Ui
    重建 E
    for
        两边都从第一个X_batch Y_batch开始训练, 
        Fi = wi - Vi
        重建F
        Y* = -i*EBj * Fi + XBi * Fi + EB * wi + ZBi
        DBi = Y*Bi - YBi

        F'Bi = DBi - V'Bi
        重建F'

        delta = i * EBT * F'B + XTBi * F'j + EBT * DBi + Z'Bi
        截断
        wi = wi - alpha/B (delta)
    重建w

    Now: 
    修改_generate_batch_data_for_localparts
    循环加入V取第j列的操作(需要核对第j列是否是第j个batch)
    F
    reconstruction
    Y...

    现在的问题: wx出来的值不对 和 Y 差的远了

    """
    

    def _generate_batch_data_and_triples(self, E, batch_size):
        # for two parties in secureML model to generate the batches
        # E X Y (V V',校对在迭代过程中,列序号能对应batch序号即可) Z Z'
        flag = "offline"
        X_batch_listA = []
        X_batch_listB = []
        y_batch_listA = []
        y_batch_listB = []
        E_batch_list = []

        U0_batch_list = []
        U1_batch_list = []

        Z0_batch_list = []
        Z1_batch_list = []
        
        V_p0_batch_list = []
        V_p1_batch_list = []
        # V_p_batch_list = []
        Z_p0_batch_list = []
        Z_p1_batch_list = []

        # Z_p0_batch_list = []
        # Z_p1_batch_list = []
        # self.indice = math.floor(ratio * X.shape[1]) # 纵向划分数据集，位于label一侧的特征数量
        
        for i in range(len(self.Y_A) // batch_size):
            # X_tmpA = X1[i * batch_size : i * batch_size + batch_size, :]
            X_batch_listA.append(self.local_matrix_A[i * batch_size : i * batch_size + batch_size, :])
            X_batch_listB.append(self.local_matrix_B[i * batch_size : i * batch_size + batch_size, :])
            y_batch_listA.append(self.Y_A[i * batch_size : i * batch_size + batch_size])
            y_batch_listB.append(self.Y_B[i * batch_size : i * batch_size + batch_size])
            
            # E, Z0, Z1, Z'0, Z'1, batch_num
            E_batch_list.append(E[i * batch_size : i * batch_size + batch_size])
            
            Z0_batch_list.append(self.Z0[i * batch_size : i * batch_size + batch_size])
            Z1_batch_list.append(self.Z1[i * batch_size : i * batch_size + batch_size])
            U0_batch_list.append(self.U0[i * batch_size : i * batch_size + batch_size])
            U1_batch_list.append(self.U1[i * batch_size : i * batch_size + batch_size])
            self.batch_num.append(batch_size)

            V_p_batch = np.random.rand(batch_size)
            V_p0_, V_p1_ = self.secret_share_vector_plaintext(V_p_batch, flag)
            V_p0_batch_list.append(V_p0_)   # 生成 V'[i]_0 , 用于mask Y*-Y
            V_p1_batch_list.append(V_p1_)   # 生成 V'[i]_1 , 用于mask Y*-Y

            Z_p_batch = np.dot(self.U[i * batch_size : i * batch_size + batch_size].T, V_p_batch)
            Z_p0_batch, Z_p1_batch = self.secret_share_vector_plaintext(Z_p_batch, flag)

            Z_p0_batch_list.append(Z_p0_batch)
            Z_p1_batch_list.append(Z_p1_batch)

        if (len(self.Y_A) % batch_size > 0):
            # X_tmpA, X_tmpB = np.hsplit(X[len(y) // batch_size * batch_size:, :], [self.indice])
            # X_batch_list.append(X[len(y) // batch_size * batch_size:, :])
            X_batch_listA.append(self.local_matrix_A[len(self.Y_A) // batch_size * batch_size:, :])
            X_batch_listB.append(self.local_matrix_B[len(self.Y_A) // batch_size * batch_size:, :])
            y_batch_listA.append(self.Y_A[len(self.Y_A) // batch_size * batch_size:])
            y_batch_listB.append(self.Y_B[len(self.Y_A) // batch_size * batch_size:])
            
            # E, Z0, Z1, Z'0, Z'1, batch_num
            E_batch_list.append(E[len(self.Y_A) // batch_size * batch_size:])

            Z0_batch_list.append(self.Z0[len(self.Y_A) // batch_size * batch_size:])
            Z1_batch_list.append(self.Z1[len(self.Y_A) // batch_size * batch_size:])
            U0_batch_list.append(self.U0[len(self.Y_A) // batch_size * batch_size:])
            U1_batch_list.append(self.U1[len(self.Y_A) // batch_size * batch_size:])
            self.batch_num.append(len(self.Y_A) % batch_size)

            """ 生成V'和Z'并且SS到两方 """
            V_p_batch = np.random.rand(len(self.Y_A) % batch_size)
            V_p0_, V_p1_ = self.secret_share_vector_plaintext(V_p_batch, flag)
            V_p0_batch_list.append(V_p0_)   # 生成 V'[i]_0 , 用于mask Y*-Y
            V_p1_batch_list.append(V_p1_)   # 生成 V'[i]_1 , 用于mask Y*-Y

            Z_p_batch = np.dot(self.U[len(self.Y_A) // batch_size * batch_size:].T, V_p_batch)
            Z_p0_batch, Z_p1_batch = self.secret_share_vector_plaintext(Z_p_batch, flag)

            Z_p0_batch_list.append(Z_p0_batch)
            Z_p1_batch_list.append(Z_p1_batch)


        print("Batch data generation: \033[32mOK\033[0m")
        return X_batch_listA, X_batch_listB, y_batch_listA, y_batch_listB, E_batch_list, Z0_batch_list, Z1_batch_list, U0_batch_list, U1_batch_list, V_p0_batch_list, V_p1_batch_list, Z_p0_batch_list, Z_p1_batch_list # listA——持有label一侧，较多样本; listB——无label一侧
    

    


    def fit_model_secure_distributed_input(self, X_batch_listA, X_batch_listB, y_batch_listA, y_batch_listB, 
                                           E_batch_list, Z0_batch_list, Z1_batch_list, U0_batch_list, U1_batch_list, V_p0_batch_list, V_p1_batch_list, Z_p0_batch_list, Z_p1_batch_list, 
                                           instances_count, converge_ondecide):
        """
        Input: 
        --------
            train data(vertically partition) 训练数据:  
                        Batch data of Party A, 
                        Batch data of Party B
            label (Secret shared) 训练数据标签:         
                        Batch y of Party A, 
                        Batch y of Party B
            Masked matrix (E = X - U):                 
                        E_batch_list             
            Triples (Z0 Z1 from Z's share, U0/U1 from U's share) 三元组: 
                        Z0_batch_list, Z1_batch_list, 
                        U0_batch_list, U1_batch_list
            instances_count: 样本总量
            converge_ondecide: 是否将loss之差作为停止准则, on则作为准则之一, off则只将max_iteration作为准则

        Update: 
        --------
            self.model_weights 模型参数
        """
        # indice_littleside 用于划分权重, 得到特征数值较小的那一部分的权重-或者左侧 默认X1一侧
        # print("ratio: ", self.ratio)

        self.n_iteration = 1
        self.loss_history = []

        # print("[CHECK] weight: ", self.weightA, self.weightB)
        self.weightA = self.weightA.reshape(-1, 1)
        self.weightB = self.weightB.reshape(-1, 1)

        ############################
        file = open(self.logname, mode='a+') #  写入记录
        time_start_training_epoch = time.time()
        ############################
        
        # print("[Hint] Training model...")
        while self.n_iteration <= self.max_iter:    # 这里的准确叫法是: epoch, iteration对应的是batch data train
            time_start_training = time.time()
            loss_list = []
            batch_label_A = None
            batch_label_B = None
            for batch_dataA, batch_dataB, batch_label_A, batch_label_B, batch_E, batch_Z0, batch_Z1, batch_V_p0, batch_V_p1, batch_Z_p0, batch_Z_p1, batch_num in zip(X_batch_listA, X_batch_listB, y_batch_listA, y_batch_listB, 
                                                                         E_batch_list, Z0_batch_list, Z1_batch_list, V_p0_batch_list, V_p1_batch_list, Z_p0_batch_list, Z_p1_batch_list, self.batch_num):
                ############################
                # file.write("batch " + str(test) + "\n")
                ############################
                
                batch_label_A = batch_label_A.reshape(-1, 1)
                batch_label_B = batch_label_B.reshape(-1, 1)

                j = 0
                # ?
                # print("[CHECK] self.V0 self.V1 shape: ", self.V0.shape, self.V1.shape)
                # import sys
                # sys.exit(0)
                # print("weightA self.Voj, shape: ", self.weightA.shape, self.V0[:,j].reshape(-1, 1).shape)
                batch_F0 = self.weightA - self.V0[:,j].reshape(-1, 1)
                batch_F1 = self.weightB - self.V1[:,j].reshape(-1, 1)
                batch_F = self.reconstruct(batch_F0, batch_F1)
                # print("batch_F shape: ", batch_F.shape, batch_F0.shape, batch_F1.shape, self.weightA.shape, self.weightB.shape, self.V0[:,j].reshape(-1, 1).shape)

                # compute the predict Y*
                Y_predictA = np.dot(batch_dataA, batch_F) + np.dot(batch_E, self.weightA) + batch_Z0[:,j].reshape(-1, 1)
                Y_predictB = np.dot(batch_dataB, batch_F) + np.dot(batch_E, self.weightB) + batch_Z1[:,j].reshape(-1, 1) + -1 * np.dot(batch_E, batch_F)
                # print("shape: ", np.dot(batch_dataA, batch_F).shape, np.dot(batch_E, self.weightA).shape, batch_Z0[:,j].reshape(-1, 1).shape)
                # print("shape: ", np.dot(batch_dataB, batch_F).shape, np.dot(batch_E, self.weightB).shape, batch_Z1[:,j].reshape(-1, 1).shape, np.dot(batch_E, batch_F).shape)
                Y_predictA = self._compute_sigmoid(Y_predictA)
                Y_predictB = self._compute_sigmoid_dual_distributed(Y_predictB)


                # compute the difference
                # print("Y_predictA shape: ", Y_predictA.shape)
                # print("Y_predictB shape: ", Y_predictB.shape)
                # print("batch_label_A shape: ", batch_label_A.shape)

                batch_D0 = Y_predictA - batch_label_A
                batch_D1 = Y_predictB - batch_label_B

                # backward
                batch_Fp0 = batch_D0 - batch_V_p0.reshape(-1, 1)
                batch_Fp1 = batch_D1 - batch_V_p1.reshape(-1, 1)
                batch_Fp = self.reconstruct(batch_Fp0, batch_Fp1)
                # print("batch_Fp shape: ", batch_Fp.shape)
                
                # print("shape: ", np.dot(batch_dataA.T, batch_Fp).shape, np.dot(batch_E.T, batch_D0).shape, batch_Z_p0.reshape(-1, 1).shape)
                delta0 = np.dot(batch_dataA.T, batch_Fp) + np.dot(batch_E.T, batch_D0) + batch_Z_p0.reshape(-1, 1)
                delta1 = np.dot(batch_dataB.T, batch_Fp) + np.dot(batch_E.T, batch_D1) + batch_Z_p1.reshape(-1, 1) + -1 * np.dot(batch_E.T, batch_Fp)
                
                # truncates
                # ......

                # update
                # print("delta0 shape: ", delta0.shape)
                # print("delta1 shape: ", delta1.shape)
                self.weightA = self.weightA - self.alpha / batch_num * (delta0) -  self.lambda_para * self.alpha * self.weightA / batch_num
                self.weightB = self.weightB - self.alpha / batch_num * (delta1) -  self.lambda_para * self.alpha * self.weightB / batch_num
                # print("[CHECK] weight shape: ", self.weightA.shape, self.weightB.shape)

                j = j + 1

                # compute loss
                # print("computing loss ...")
                batch_loss = self.secure_distributed_compute_loss_cross_entropy(label = batch_label_A + batch_label_B, 
                                                                        Y_predictA=Y_predictA, Y_predictB=Y_predictB, batch_num = batch_num)
                loss_list.append(batch_loss)

            # 打乱数据集的batch
            X_batch_listA, X_batch_listB, y_batch_listA, y_batch_listB, E_batch_list, Z0_batch_list, Z1_batch_list, Z_p0_batch_list, Z_p1_batch_list, self.batch_num = self.shuffle_lists(X_batch_listA, X_batch_listB, y_batch_listA, y_batch_listB, 
                                                                                E_batch_list, Z0_batch_list, Z1_batch_list, Z_p0_batch_list, Z_p1_batch_list, self.batch_num)
            # X_batch_listA, X_batch_listB, y_batch_listA, y_batch_listB, E_batch_list, Z0_batch_list, Z1_batch_list, Z_p0_batch_list, Z_p1_batch_list = self.shuffle_distributed_data(X_batch_listA, X_batch_listB, y_batch_listA, y_batch_listB, 
            #                                                                     E_batch_list, Z0_batch_list, Z1_batch_list, Z_p0_batch_list, Z_p1_batch_list)
            
            ## 计算 sum loss
            loss = np.sum(loss_list) / instances_count
            print("\rEpoch {}, batch sum loss: {}".format(self.n_iteration, loss), end = '')
            

            # time_end_training = time.time()
            # epoch_Online_Calculate_timeAccount = time_end_training - time_start_training_epoch # 计算时间
            # epoch_Online_Communicate_timeAccount = self.online_comm_time_account  # 通信时间
            # # offline time 不同epoch应该是相同的
            # epoch_Online_timeAccounting = epoch_Online_Calculate_timeAccount + epoch_Online_Communicate_timeAccount
            # print("self.online_comm_time_account: ", self.online_comm_time_account)
            # print("epoch_Online_Calculate_timeAccount: ", epoch_Online_Calculate_timeAccount)
            # print("epoch_Online_timeAccounting: ", epoch_Online_timeAccounting)
            # sys.exit()

            
            ############################
            time_end_training = time.time()
            # print(" Time: " + str(time_end_training-time_start_training) + "s\n")
            if self.ovr == "bin":
                file.write("Epoch {}, batch sum loss: {}".format(self.n_iteration, loss))
                file.write(" Time: " + str(time_end_training-time_start_training) + "s\n")
            # self.file.close()
            ############################

            


            """ 
            intermediate result saving 
            """
            if self.ovr == "bin" and self.n_iteration in self.EPOCH_list:
                ## Time
                time_end_training = time.time()
                epoch_Online_Calculate_timeAccount = time_end_training - time_start_training_epoch # 计算时间
                epoch_Online_Communicate_timeAccount = self.online_comm_time_account  # 通信时间
                # offline time 不同epoch应该是相同的
                epoch_Online_timeAccounting = epoch_Online_Calculate_timeAccount + epoch_Online_Communicate_timeAccount

                ## save Model and Time
                self.modelWeight_and_Time_List.update({ str(self.n_iteration): [self.weightA + self.weightB, epoch_Online_timeAccounting, self.online_comm_time_account] })

            elif self.ovr == "ovr" and self.n_iteration in self.EPOCH_list:
                ## Time
                time_end_training = time.time()
                epoch_Online_Calculate_timeAccount = time_end_training - time_start_training_epoch # 计算时间
                epoch_Online_Communicate_timeAccount = self.online_comm_time_account  # 通信时间
                # offline time 不同epoch应该是相同的
                epoch_Online_timeAccounting = epoch_Online_Calculate_timeAccount + epoch_Online_Communicate_timeAccount

                # epoch: [weight, Online_time_total, Online_time_comm]
                self.OVRModel_X.update({ str(self.n_iteration): [self.weightA + self.weightB, epoch_Online_timeAccounting, self.online_comm_time_account] })
                



            ## 判断是否停止
            self.is_converged = self.check_converge_by_loss(loss)


            # # 用于server调试
            # if(loss >= 50 or np.isnan(loss) or np.isinf(loss)):
            #     file.write("Epoch num: {}, last epoch loss: {}".format(self.n_iteration, loss))
            #     file.write("!!!!!!!!!!!overflow!!!!!!!!")


            #     alphaa = 0.9 * self.alpha
            #     cmd1 = "nohup python3 -u secureML_mock_server_OVR_upgrade_final_exp_tryovr.py -d DailySports -p 37 -m sketch -a "
            #     method = self.kernel_method
            #     cmd2 = " -k 2048 -c 4 -b 2 -o ovr -r mm -l off -s linear -al "
            #     cmd3 = " -lm 0 -i 200 -t "
            #     cmd4 = " -e 201 -f &"
            #     cmd_string = cmd1 + method + cmd2 + str(alphaa) + cmd3 + str(self.batch_size) + cmd4
            #     os.system(cmd_string)
            #     time.sleep(1.5)
            #     sys.exit()
            
            
            
            # if self.is_converged or (self.n_iteration == self.max_iter):
            if (self.is_converged and converge_ondecide == "on")  or (self.n_iteration == self.max_iter):
                if self.ratio is not None: 
                    self.model_weights = self.weightA + self.weightB
                    # print("self.model_weights: ", self.model_weights)

                    """ 第i个类别的记录: 运行时间 """
                    time_end_training = time.time()
                    if self.ovr == "ovr":
                        file.write("Epoch num: {}, last epoch loss: {}".format(self.n_iteration, loss))
                        file.write(" Epoch Total Time: " + str(time_end_training-time_start_training_epoch) + "s\n")
                break

            self.n_iteration += 1


    

    def Binary_Secure_Classifier(self, X_trainA, X_trainB, Y_train, instances_count, feature_count, indice_littleside, converge_ondecide):
        """ 二分类 """
        # indice_littleside 用于划分权重, 得到特征数值较小的那一部分的权重-或者左侧 默认X1一侧

        offline_time_start = time.time()

        print("ratio: ", self.ratio)
        self.indice = indice_littleside

        # generate shared data and labels for two parties
        self.secretSharing_Data_and_Labels(X_trainA, X_trainB, Y_train)
        # label: self.Y_A self.Y_B
        # data: self.local_matrix_A self.local_matrix_B

        # split the model weight according to data distribution
        self.weightA = self.model_weights
        self.weightB = self.model_weights

        # generate triples: U V Z V' Z'
        import math
        t = int(math.ceil(instances_count/self.batch_size))
        print("t: ", t)
        self.generate_UVZV_Z_multTriplets_beaver_triplets(instances_count, feature_count, 
                                                          t, self.batch_size)
        # Mask X0 X1 and reconstruct E
        E0 = self.local_matrix_A - self.U0
        E1 = self.local_matrix_B - self.U1
        E = self.reconstruct(E0, E1)

        # generate batch data:
        X_batch_listA, X_batch_listB, y_batch_listA, y_batch_listB, E_batch_list, Z0_batch_list, Z1_batch_list, U0_batch_list, U1_batch_list, V_p0_batch_list, V_p1_batch_list, Z_p0_batch_list, Z_p1_batch_list = self._generate_batch_data_and_triples(E, self.batch_size)
        # X_batch_listA, X_batch_listB, y_batch_listA, y_batch_listB, E_batch_list, Z0_batch_list, Z1_batch_list, Zp1_batch_list, Zp2_batch_list = self._generate_batch_data_and_triples(E, self.batch_size)
        # 这些batch data可以在过程中计算得到: Z_batch_list, Z_p_batch_list (算了一起生成吧)

        offline_time_end = time.time()
        file = open(self.logname, mode='a+') #  写入记录
        file.write("\n =================== # Training Offline Phase # =================== ")
        self.offline_calculate_time += offline_time_end-offline_time_start
        file.write("\nOffline Total Time: " + str(self.offline_calculate_time + self.offline_comm_time_account) + "s")
        file.write("\nOffline Communication Time: " + str(self.offline_comm_time_account) + "s")
        file.write("\n ================= # Training Offline Phase End # ================== \n")
        

        self.fit_model_secure_distributed_input(X_batch_listA, X_batch_listB, y_batch_listA, y_batch_listB, E_batch_list, Z0_batch_list, Z1_batch_list, U0_batch_list, U1_batch_list, V_p0_batch_list, V_p1_batch_list, Z_p0_batch_list, Z_p1_batch_list, instances_count, converge_ondecide)



    def predict_distributed_OVR(self, x_test1, x_test2):
        x_test = np.hstack((x_test1, x_test2))
        if self.data_tag == 'sparse':
            z = x_test.dot(self.model_weights.T)    # np.array类型（此处其实需要严谨一点，避免数据类型不清晰影响后续运算）
            if not isinstance(z, np.ndarray):
                z = z.toarray()
        elif self.data_tag == None:
            self.model_weights = self.model_weights.reshape(-1, 1)
            z = np.dot(x_test, self.model_weights)

        
        self.time_counting_model_inference(self.model_weights) # 前向重建F的通信量 (TODO)
        self.time_counting_model_inference(self.model_weights) # 前向重建F的通信量

        y = self._compute_sigmoid(z)

        return y.reshape(1, -1) # list(y.reshape((1, -1)))


    def y_update_OVR(self, Y_train, batch_size):
        """ 依据OVR多分类原理, 依次将某一类设置为正样本, 其他类为负样本 """
        local_Y, share_Y = self.secret_share_vector_plaintext(Y_train)
        self.Y_A = local_Y
        self.Y_B = share_Y

        y_batch_listA = []
        y_batch_listB = []

        for i in range(len(self.Y_A) // batch_size):
            # X_tmpA = X1[i * batch_size : i * batch_size + batch_size, :]
            y_batch_listA.append(self.Y_A[i * batch_size : i * batch_size + batch_size])
            y_batch_listB.append(self.Y_B[i * batch_size : i * batch_size + batch_size])
            # self.batch_num.append(batch_size)

        if (len(self.Y_A) % batch_size > 0):
            y_batch_listA.append(self.Y_A[len(self.Y_A) // batch_size * batch_size:])
            y_batch_listB.append(self.Y_B[len(self.Y_A) // batch_size * batch_size:])
            # self.batch_num.append(len(self.Y_A) % batch_size)

        # print("Batch data generation: \033[32mOK\033[0m")
        return y_batch_listA, y_batch_listB


    def OneVsRest_Secure_Classifier(self, X_train1, X_train2, X_test1, X_test2, Y_train, Y_test, converge_ondecide):
        """
        OVR: one vs rest 多分类
        """
        offline_time_start = time.time()

        indice_littleside = X_train1.shape[1]
        self.indice = X_train1.shape[1]
        instances_count = X_train1.shape[0]
        label_lst = list(set(Y_train))   # 多分类的所有标签值集合
        print('数据集标签值集合: ', label_lst)
        prob_lst = []                    # 存储每个二分类模型的预测概率值

        """ OVR Model Training """
        """ batch 数据生成 """
        feature_count = X_train1.shape[1]+X_train2.shape[1]
        self.indice = indice_littleside
        # generate shared data and labels for two parties
        self.secretSharing_Data_and_Labels(X_train1, X_train2, Y_train)
        # label: self.Y_A self.Y_B
        # data: self.local_matrix_A self.local_matrix_B

        # split the model weight according to data distribution
        self.weightA = self.model_weights
        self.weightB = self.model_weights

        # generate triples: U V Z V' Z'
        import math
        t = int(math.ceil(instances_count/self.batch_size))
        self.generate_UVZV_Z_multTriplets_beaver_triplets(instances_count, feature_count, 
                                                          t, self.batch_size)
        # Mask X0 X1 and reconstruct E
        E0 = self.local_matrix_A - self.U0
        E1 = self.local_matrix_B - self.U1
        E = self.reconstruct(E0, E1)

        # generate batch data:
        # X_batch_listA, X_batch_listB, y_batch_listA, y_batch_listB, E_batch_list, Z0_batch_list, Z1_batch_list, U0_batch_list, U1_batch_list = self._generate_batch_data_and_triples(E, self.batch_size)
        X_batch_listA, X_batch_listB, y_batch_listA, y_batch_listB, E_batch_list, Z0_batch_list, Z1_batch_list, U0_batch_list, U1_batch_list, V_p0_batch_list, V_p1_batch_list, Z_p0_batch_list, Z_p1_batch_list = self._generate_batch_data_and_triples(E, self.batch_size)
        # X_batch_listA, X_batch_listB, y_batch_listA, y_batch_listB, E_batch_list, Z0_batch_list, Z1_batch_list, Zp1_batch_list, Zp2_batch_list = self._generate_batch_data_and_triples(E, self.batch_size)
        # 这些batch data可以在过程中计算得到: Z_batch_list, Z_p_batch_list (算了一起生成吧)

        offline_time_end = time.time()
        file = open(self.logname, mode='a+') #  写入记录
        file.write("\n =================== # Training Offline Phase # =================== ")
        self.offline_calculate_time += offline_time_end-offline_time_start
        file.write("\nOffline Total Time: " + str(self.offline_calculate_time + self.offline_comm_time_account) + "s")
        file.write("\nOffline Communication Time: " + str(self.offline_comm_time_account) + "s")
        file.write("\n ================= # Training Offline Phase End # ================== \n")

        ## 时间统计
        x_test = np.hstack((X_test1, X_test2))
        self.time_counting_model_inference(x_test) # 假装加上mask矩阵E的通信量
        self.time_counting_model_inference(x_test) # 数据分享的通信量

        # self.OVRModel_Agg

        for i in range(len(label_lst)):
            # 转换标签值为二分类标签值
            pos_label = label_lst[i]                                        # 选定正样本的标签
            file = open(self.logname, mode='a+') #  写入记录
            # print("Label: ", pos_label)
            file.write("Label {}".format(pos_label))

            # def label_reset_OVR(arr):
            #     """ 依次将标签i设置为正样本, 其他为负样本 """
            #     # global pos_label
            #     return np.where(arr == pos_label, 1, 0)
            
            # y_batch_list = list(map(label_reset_OVR, Y_train))

            Y_train_new = np.where(Y_train == pos_label, 1, 0)

            y_batch_listA, y_batch_listB = self.y_update_OVR(Y_train_new, self.batch_size) # 因为把batch数据生成移到外面了, 每次要重新生成y标签的batch
            
            # Y_train_new = np.where(Y_train == pos_label, 1, 0)              # 满足条件则为正样本1，否则为负样本0
            # Y_test_new = np.where(Y_test == pos_label, 1, 0)
            # print(Y_train_new)
            # self.fit_model_secure_distributed_input(X_batch_listA, X_batch_listB, y_batch_listA, y_batch_listB, 
            #                                     E_batch_list, Z0_batch_list, Z1_batch_list, U0_batch_list, U1_batch_list,
            #                                     instances_count, converge_ondecide)

            self.weightA = np.zeros(X_train1.shape[1]+X_train2.shape[1]).reshape(-1, 1)
            self.weightB = np.zeros(X_train1.shape[1]+X_train2.shape[1]).reshape(-1, 1)
            self.model_weights = np.zeros(X_train1.shape[1]+X_train2.shape[1]).reshape(-1, 1)

            self.OVRModel_X = dict() # 用于临时存储某个label下的多个epoch对应的二分类模型参数和时间
            
            self.fit_model_secure_distributed_input(X_batch_listA, X_batch_listB, y_batch_listA, y_batch_listB, 
                                                    E_batch_list, Z0_batch_list, Z1_batch_list, U0_batch_list, U1_batch_list, 
                                                    V_p0_batch_list, V_p1_batch_list, Z_p0_batch_list, Z_p1_batch_list, 
                                                    instances_count, converge_ondecide)
            
            self.OVRModel_Agg.update( {pos_label: self.OVRModel_X} ) # 存储
            
            """ 第二个开始, submodel的时间统计从0开始, 一开始的数据分享的通信时间被包含每种epoch记录的第一个submodel的时间记录中. """
            self.online_comm_time_account = 0
            
            if self.EPOCH_list is None: # 不适合 epoch变化跑实验
                prob = self.predict_distributed_OVR(X_test1, X_test2)   # 第i个二分类模型在测试数据集上，每个样本取正标签的概率（用决策函数值作为概率值）
                prob = np.where(prob > 0, prob, 0).flatten()
                prob_lst.append(prob.tolist())
            # print(prob_lst)
        
        if self.EPOCH_list is not None: # epoch变化跑实验
            """ 模型预测 """
            self.predict_forEpochs_OVR(X_test1, X_test2, label_lst)
            # end

        else:
            # 多分类模型预测
            print(np.shape(prob_lst))
            y_predict = []                      # 存储多分类的预测标签值
            prob_array = np.asarray(prob_lst).T   # (n_samples, n_classes)
            print(prob_array.shape)
            print(type(prob_array))
            print(type(prob_array[0]))
            print(type(prob_array[0][0]))

            for i in range(len(Y_test)):
                temp = list(prob_array[i])
                index = temp.index(max(temp))
                # print(index)
                y_predict.append(label_lst[index])
            # print(y_predict)
            # 模型预测准确率
            self.score = 0
            for i in range(len(y_predict)):
                if y_predict[i] == Y_test[i]:
                    self.score += 1
                else:
                    pass

            print("score: ", self.score)
            self.total_num = len(y_predict)
            print("len(y): ", self.total_num)
            self.accuracy = float(self.score)/float(len(y_predict))
            print("\nPredict precision: ", self.accuracy)




    def predict_forEpochs_OVR(self, X_test1, X_test2, label_lst):
        """ Epochs OVR 统计不同Epoch下的模型结果 """
        file = open(self.logname, mode='a+') #  写入记录

        inference_time_base_SSdata = self.inference_time_account
        self.inference_time_account = 0

        accuracy_list = []
        total_Online_commtime_list = []
        total_Online_time_list = []
        total_time_cost_list = []
        inference_time_total_list = []


        for obj in self.EPOCH_list:
            # epoch = obj 时的结果
            total_Online_time = 0
            total_Online_commtime = 0
            prob_lst = []
            for submodel_iter in range(len(label_lst)): ## 1, len(self.OVRModel_Agg)+1):
                # 第submodel_iter个子模型的结果
                pos_label = label_lst[submodel_iter]
                subModelWeight = self.OVRModel_Agg[pos_label][str(obj)][0]  # 子模型pos_label (epoch = obj下的结果)
                total_Online_time += self.OVRModel_Agg[pos_label][str(obj)][1]
                total_Online_commtime += self.OVRModel_Agg[pos_label][str(obj)][2]

                
                prob = self.predict_base_OVR(X_test1, X_test2, subModelWeight)   # 第i个二分类模型在测试数据集上，每个样本取正标签的概率（用决策函数值作为概率值）
                prob = np.where(prob > 0, prob, 0).flatten()
                prob_lst.append(prob.tolist())

            accuracy, score, total_num = self.predict_MAX_OVR(prob_lst, label_lst)

            # file.write("\n\n======================================== ###   Epoch {}   ### ========================================".format(obj))
            # file.write("\n# ================== #   Train Time   # ================== #")
            # file.write("\nSecureMLModel online comm_time account: {}s".format(total_Online_commtime))
            # file.write("\nSecureMLModel Total online time account: {}s".format(total_Online_time))
            # # 总开销(online和offline的计算和通信时间):
            # file.write("\nTotal time cost(online+offline, comm+cal): {}s".format(total_Online_time + self.offline_calculate_time + self.offline_comm_time_account))
            
            # # file = open(self.logname, mode='a+') #  写入记录
            # file.write("\n# =================== #  Test Model  # =================== #")
            # file.write("\nscore: {}".format(score))
            # file.write("\nlen(y): {}".format(total_num))
            # file.write("\nPredict precision: {}".format(accuracy))

            # file.write("\n# ================== #   Inference Time   # ================== #")
            # file.write("\nSecureMLModel inference_time SS data: {}s".format(inference_time_base_SSdata))
            # file.write("\nSecureMLModel inference_time account(total): {}s".format(self.inference_time_account + inference_time_base_SSdata))

            accuracy_list.append(accuracy)
            total_Online_commtime_list.append(total_Online_commtime)
            total_Online_time_list.append(total_Online_time)
            total_time_cost_list.append(total_Online_time + self.offline_calculate_time + self.offline_comm_time_account)
            inference_time_total_list.append(self.inference_time_account + inference_time_base_SSdata)

        file.write("\n\nConclusion:")
        file.write("\nEpoch list: {}".format(self.EPOCH_list))
        file.write("\nSecureML Accuracy List: {}".format(accuracy_list))
        file.write("\nSecureML total_Online_commtime List: {}".format(total_Online_commtime_list))
        file.write("\nSecureML total_Online_time List: {}".format(total_Online_time_list))
        file.write("\nSecureML total_time_cost List: {}".format(total_time_cost_list))
        file.write("\nSecureML inference_time_total List: {}".format(inference_time_total_list))

        file.close()
        print("End.")


    def predict_base_OVR(self, X_test1, X_test2, subModelWeight):
        x_test = np.hstack((X_test1, X_test2))
        # if self.data_tag == 'sparse':
        #     z = x_test.dot(self.model_weights.T)    # np.array类型（此处其实需要严谨一点，避免数据类型不清晰影响后续运算）
        #     if not isinstance(z, np.ndarray):
        #         z = z.toarray()
        # elif self.data_tag == None:
        self.model_weights = subModelWeight.reshape(-1, 1)
        z = np.dot(x_test, self.model_weights)

        # self.time_counting_model_inference(x_test) # 假装加上mask矩阵E的通信量
        # self.time_counting_model_inference(x_test) # 数据分享的通信量
        self.time_counting_model_inference(self.model_weights) # 前向重建F的通信量
        self.time_counting_model_inference(self.model_weights) # 前向重建F的通信量

        y = self._compute_sigmoid(z)

        return y.reshape(1, -1) # list(y.reshape((1, -1)))


    def predict_MAX_OVR(self, prob_lst, label_lst):
        y_predict = []                      # 存储多分类的预测标签值
        prob_array = np.asarray(prob_lst).T   # (n_samples, n_classes)

        for i in range(len(Y_test)):
            temp = list(prob_array[i])
            index = temp.index(max(temp))
            # print(index)
            y_predict.append(label_lst[index])
        # print(y_predict)
        # 模型预测准确率
        score = 0
        for i in range(len(y_predict)):
            if y_predict[i] == Y_test[i]:
                score += 1
            else:
                pass

        print("score: ", score)
        total_num = len(y_predict)
        print("len(y): ", total_num)
        accuracy = float(score)/float(total_num)
        print("\nPredict precision: ", accuracy)

        return accuracy, score, total_num

        







    def predict_distributed(self, x_test1, x_test2, y_test):
        # z = np.dot(x_test, self.model_weights.T)
        # z = x_test.dot(self.model_weights.T)
        # z = self._cal_z()
        x_test = np.hstack((x_test1, x_test2))
        if self.data_tag == 'sparse':
            z = x_test.dot(self.model_weights.T) # np.dot(features, weights.T)
        elif self.data_tag == None:
            # SecureML
            self.model_weights = self.model_weights.reshape(-1, 1)
            z = np.dot(x_test, self.model_weights)

        # self.time_counting_model_inference(x_test) # 假装加上mask矩阵E的通信量
        # self.time_counting_model_inference(x_test) # 数据分享的通信量
        # self.time_counting_model_inference(self.model_weights) # 前向重建F的通信量
        # self.time_counting_model_inference(self.model_weights) # 前向重建F的通信量

        y = self._compute_sigmoid(z)

        self.score = 0
        for i in range(len(y)):
            if y[i] >= 0.5: y[i] = 1
            else: y[i] = 0
            if y[i] == y_test[i]:
                self.score += 1
            else:
                pass

        print("score: ", self.score)
        self.total_num = len(y)
        print("len(y): ", self.total_num)
        self.accuracy = float(self.score)/float(len(y))
        print("\nPredict precision: ", self.accuracy)


    def predict_base(self, x_test1, x_test2, y_test, weight):
        """
        多个epoch不同weight的预测
        """
        # z = np.dot(x_test, self.model_weights.T)
        # z = x_test.dot(self.model_weights.T)
        # z = self._cal_z()
        x_test = np.hstack((x_test1, x_test2))
        weight = weight.reshape(-1, 1)
        z = np.dot(x_test, weight)

        self.time_counting_model_inference(x_test) # 假装加上mask矩阵E的通信量
        self.time_counting_model_inference(x_test) # 数据分享的通信量
        self.time_counting_model_inference(weight) # 前向重建F的通信量
        self.time_counting_model_inference(weight) # 前向重建F的通信量

        y = self._compute_sigmoid(z)

        score = 0
        for i in range(len(y)):
            if y[i] >= 0.5: y[i] = 1
            else: y[i] = 0
            if y[i] == y_test[i]:
                score += 1
            else:
                pass

        print("score: ", score)
        total_num = len(y)
        print("len(y): ", total_num)
        accuracy = float(score)/float(total_num)
        print("Predict precision: ", accuracy)

        return accuracy, score, total_num


    def predict_forEpochs_Bin(self, x_test1, x_test2, y_test):
        """
        实验中对多个epoch存储的model weight预测
        """
        file = open(self.logname, mode='a+') #  写入记录

        x_test = np.hstack((x_test1, x_test2))

        self.time_counting_model_inference(x_test) # 假装加上mask矩阵E的通信量
        self.time_counting_model_inference(x_test) # 假装加上mask矩阵E的通信量
        self.time_counting_model_inference(x_test1) # 分享数据的通信量
        self.time_counting_model_inference(x_test2) # 分享数据的通信量

        inference_time_base_SSdata = self.inference_time_account
        self.inference_time_account = 0

        accuracy_list = []
        total_Online_commtime_list = []
        total_Online_time_list = []
        total_time_cost_list = []
        inference_time_total_list = []

        for obj in self.EPOCH_list:
            weight = self.modelWeight_and_Time_List[str(obj)][0]  # [0]: model_weight, [1]: time(Total online time)
            total_Online_time = self.modelWeight_and_Time_List[str(obj)][1]
            total_Online_commtime = self.modelWeight_and_Time_List[str(obj)][2]
            print("Epoch: {}".format(obj))
            accuracy, score, total_num = self.predict_base(x_test1, x_test2, y_test, weight)


            # file.write("\n\n======================================== ###   Epoch {}   ### ========================================".format(obj))
            # file.write("\n# ================== #   Train Time   # ================== #")
            # file.write("\nSecureMLModel online comm_time account: {}s".format(total_Online_commtime))
            # file.write("\nSecureMLModel Total online time account: {}s".format(total_Online_time))
            # # 总开销(online和offline的计算和通信时间):
            # file.write("\nTotal time cost(online+offline, comm+cal): {}s".format(total_Online_time + self.offline_calculate_time + self.offline_comm_time_account))
            
            # # file = open(self.logname, mode='a+') #  写入记录
            # file.write("\n# =================== #  Test Model  # =================== #")
            # file.write("\nscore: {}".format(score))
            # file.write("\nlen(y): {}".format(total_num))
            # file.write("\nPredict precision: {}".format(accuracy))

            # file.write("\n# ================== #   Inference Time   # ================== #")
            # file.write("\nSecureMLModel inference_time account: {}s".format(self.inference_time_account))

            accuracy_list.append(accuracy)
            total_Online_commtime_list.append(total_Online_commtime)
            total_Online_time_list.append(total_Online_time)
            total_time_cost_list.append(total_Online_time + self.offline_calculate_time + self.offline_comm_time_account)
            inference_time_total_list.append(self.inference_time_account + inference_time_base_SSdata)

            self.inference_time_account = 0

        file.write("\n\nConclusion:")
        file.write("\nEpoch list: {}".format(self.EPOCH_list))
        file.write("\nSecureML Accuracy List: {}".format(accuracy_list))
        file.write("\nSecureML total_Online_commtime List: {}".format(total_Online_commtime_list))
        file.write("\nSecureML total_Online_time List: {}".format(total_Online_time_list))
        file.write("\nSecureML total_time_cost List: {}".format(total_time_cost_list))
        file.write("\nSecureML inference_time_total List: {}".format(inference_time_total_list))
        
        file.close()
        print("End.")





def read_distributed_data_raw_or_sketch(dataset_name, raw_or_sketch, kernel_method, portion, sampling_k, ovr, countsketch_, bbitmwhash_, scalering_raw):
    """
    Desc: 
    ----
    load data.

    Para:
    ----
    dataset_name: name of input dataset
    raw_or_sketch: input data - raw data or sketch data
    kernel_method: pminhash / 0bitcws / rff / poly
    portion: vertically partition scale
    sampling_k: sampling times(k)
    ovr:  one vs rest strategy
    scalering_raw: scalering raw data or not
    """
    ## countsketch
    from sklearn.datasets import load_svmlight_file
    from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, Normalizer, RobustScaler
    mm = MinMaxScaler()
    ss = StandardScaler()
    na = Normalizer()
    ma = MaxAbsScaler()
    rs = RobustScaler()

    main_path = PATH_DATA
    dataset_file_name = dataset_name  
    train_file_name = dataset_name + '_train.txt' 
    test_file_name = dataset_name + '_test.txt'
    # dataset_file_name = 'DailySports'  
    # train_file_name = 'DailySports_train.txt' 
    # test_file_name = 'DailySports_test.txt'
    

    """
    读取数据集的 Label: Y_train, Y_test
    """
    print("loading dataset...")
    train_data = load_svmlight_file(os.path.join(main_path, dataset_file_name, train_file_name))
    test_data = load_svmlight_file(os.path.join(main_path, dataset_file_name, test_file_name))

    Y_train = train_data[1].astype(int)
    Y_test = test_data[1].astype(int)

    """
    数据集抽取(对于部分较大数据集),
    将训练集抽取一部分
    其余数据集: 只是返回原始的训练集和训练集标签
        sketch部分:只需要抽取Y
        raw部分:需要X和Y都抽取
    返回的结果: numpy格式的X_train和Y_train (抽取后的或没有经过抽取的)
    (后面再写, 最好是采样和raw 抽取的是同一部分样本)
    """
    X_train, X_test, Y_train = dataset_selector(dataset_name, train_data, test_data, Y_train)
    # X_test, Y_test = dataset_selector(dataset_name, test_data, Y_test)
    


    """
    标签检查和重构
    """
    ##### 判断标签是(1;-1)还是 (1;0), 将-1的标签全部转化成0标签
    # if -1 in Y_train:  
    #     Y_train[Y_train == -1] = 0
    #     Y_test[Y_test == -1] = 0
    
    print("processing dataset...")
    if ovr == "bin":
        """ 二分类数据集的标签处理为: 1 0 分类 """
        if -1 in Y_train:
            # 判断标签是(1;-1)还是 (1;0), 将-1的标签全部转化成0标签
            Y_train[Y_train != 1] = 0
            Y_test[Y_test != 1] = 0


    """
    确定最终的目标数据集路径
    """
    if portion == "37": partition = 3/10
    elif portion == "28": partition = 2/10
    elif portion == "19": partition = 1/10
    elif portion == "46": partition = 4/10
    elif portion == "55": partition = 5/10
    else: raise ValueError

    """
    一个不知道怎么处理的参数: gamma_scale
    """
    gamma_scale = -1


    if raw_or_sketch == "sketch" and kernel_method in ["pminhash", "0bitcws"]:

        if countsketch_ == 0 and bbitmwhash_ == 0:
            ## 目前这里跑实验的时候, 这两个值不能同时为0, 否则没有意义
            raise ValueError("[error] b(bbitmwhash) and c(countsketch) equals to 0 at the same time!")

        """ 获取需要读取的sketch的相对路径 """
        portion_kernel_method = "portion" + portion + "_" + kernel_method
        sketch_sample = "sketch" + sampling_k

        # dataset_file_name = 'kits/portion37_pminhash/sketch1024/countsketch/'
        # train_file_name1 = 'X1_squeeze_train37_Countsketch.txt'
        # train_file_name2 = 'X2_squeeze_train37_Countsketch.txt'
        # test_file_name1 = 'X1_squeeze_test37_Countsketch.txt'
        # test_file_name2 = 'X2_squeeze_test37_Countsketch.txt'

        if kernel_method == "pminhash" and countsketch_:
            """ sketch + countsketch """
            dataset_file_name = os.path.join(dataset_name, portion_kernel_method, sketch_sample, "countsketch"+"_"+str(countsketch_))
            train_file_name1 = 'X1_squeeze_train37.txt'
            train_file_name2 = 'X2_squeeze_train37.txt'
            test_file_name1 = 'X1_squeeze_test37.txt'
            test_file_name2 = 'X2_squeeze_test37.txt'

        elif kernel_method == "0bitcws" and bbitmwhash_:
            """ sketch + bbitmwhash """
            dataset_file_name = os.path.join(dataset_name, portion_kernel_method, sketch_sample, "bbitmwhash"+"_"+str(2**bbitmwhash_))
            train_file_name1 = 'X1_squeeze_train37.txt'
            train_file_name2 = 'X2_squeeze_train37.txt'
            test_file_name1 = 'X1_squeeze_test37.txt'
            test_file_name2 = 'X2_squeeze_test37.txt'

        else:
            """ sketch only 毫无意义的入口, 我也不知道为什么保留到现在"""
            dataset_file_name = os.path.join(dataset_name, portion_kernel_method, sketch_sample)
            train_file_name1 = 'X1_train_samples.txt'
            train_file_name2 = 'X2_train_samples.txt'
            test_file_name1 = 'X1_test_samples.txt'
            test_file_name2 = 'X2_test_samples.txt'
            raise ValueError("Attempt to read some meaningless data as model training data.")

        X_train1 = np.loadtxt(os.path.join(main_path, dataset_file_name, train_file_name1), delimiter=',') #, dtype = float) # <class 'numpy.float64'>
        X_train2 = np.loadtxt(os.path.join(main_path, dataset_file_name, train_file_name2), delimiter=',') #, dtype = float)
        X_test1 = np.loadtxt(os.path.join(main_path, dataset_file_name, test_file_name1), delimiter=',') #, dtype = float)
        X_test2 = np.loadtxt(os.path.join(main_path, dataset_file_name, test_file_name2), delimiter=',') #, dtype = float)

        # print(X_train1) # [[0. 1. 0. ... 1. 0. 1.]]
        # print(type(X_train1[0][0])) # <class 'numpy.float64'>



    elif raw_or_sketch == "sketch" and kernel_method in ["rff", "poly"]:
        X_test = test_data[0].todense().A

        # 输入的X_train是原数据集(如果数据集过大, dataset_selector中会随机抽取一部分, 具体见dataset_selector函数); 
        # 如果抽取了X_train, Y_train也会对应着抽取
        X_train1, X_train2, Y_train, X_test1, X_test2, Y_test, gamma_scale = dataSketch_generator_RBFandPolyKernel(X_train, X_test, Y_train, Y_test, kernel_method, sampling_k, partition)



        
    elif raw_or_sketch == "raw":
        """ 对于 Raw data, 直接读入原始数据, 然后按照比例 portion 分成两个部分, 作为两方数据 """
        print("Try to read Raw data...")

        scalering_list = ["mm", "ss", "na", "ma", "rs"]
        if scalering_raw in scalering_list:
            """ from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, Normalizer, RobustScaler
                mm = MinMaxScaler()
                ss = StandardScaler()
                na = Normalizer()
                ma = MaxAbsScaler()
                rs = RobustScaler() 
            """
            scaler = eval(scalering_raw)
            # X_train = train_data[0].todense().A
            X_train = scaler.fit_transform(X_train)
            X_test = test_data[0].todense().A
            X_test = scaler.fit_transform(X_test)

        elif scalering_raw == "null":
            # X_train = train_data[0].todense().A
            # X_train = mm.fit_transform(X_train)
            X_test = test_data[0].todense().A
            # X_test = mm.fit_transform(X_test)

        # X_train
        k = X_train.shape[1] # 总特征数
        # partition = 3/10
        k1 = np.floor(k * partition).astype(int) # X1的特征数
        X_train1, X_train2 = X_train[:,0:k1], X_train[:,k1:]

        # X_test
        k = X_test.shape[1]
        # partition = 3/10
        k1 = np.floor(k * partition).astype(int)
        X_test1, X_test2 = X_test[:,0:k1], X_test[:,k1:]

    
    print("X_train1 type: ", type(X_train1)) # 1000 * 60
    print("X_train1 shape: ", X_train1.shape)
    print("X_train2 type: ", type(X_train2)) # 1000 * 60
    print("X_train2 shape: ", X_train2.shape)
    print("X_test1 type: ", type(X_test1)) # 1000 * 60
    print("X_test1 shape: ", X_test1.shape)
    print("X_test2 type: ", type(X_test2)) # 1000 * 60
    print("X_test2 shape: ", X_test2.shape)
    print("Y_train type: ", type(Y_train))
    print("Y_test type: ", type(Y_test))
    
    return X_train1, X_train2, Y_train, X_test1, X_test2, Y_test, gamma_scale




def logger_info(objectmodel, dataset_name, raw_or_sketch, kernel_method, portion, sampling_k, countsketch_, bbitmwhash_, gamma_scale,
                X_train1_shape, X_train2_shape, X_test1_shape, X_test2_shape, Y_train_shape, Y_test_shape, sigmoid_func, scalering_raw, converge_ondecide):

    file = open(objectmodel.logname, mode='w+') #  写入记录
    file.write("\n =================== # Dataset info # =================== ")
    file.write("\nData source: {} - {}".format(dataset_name, raw_or_sketch))
    file.write("\nFeature: {}".format(objectmodel.ovr)) # bin / ovr 二分类 或 多分类
    file.write("\nData Portion: {}".format(portion))  # ratio

    if raw_or_sketch == "sketch":
        """ sketch data info """
        file.write("\nSketching method: {}".format(kernel_method))
        file.write("\nSampling k: {}".format(sampling_k))
        # if countsketch_: file.write("\nUsing Counsketch: c = {}".format(countsketch_))

        if kernel_method == "pminhash": file.write("\nUsing Counsketch: c = {}".format(countsketch_))
        elif kernel_method == "0bitcws": file.write("\nUsing bbit-minwise-hash: b = {}, 2^b = {}".format(bbitmwhash_, 2**bbitmwhash_))
        elif kernel_method == "rff": file.write("\nUsing rff: gamma = {}".format(gamma_scale))
        elif kernel_method == "poly": file.write("\nUsing poly(TS): gamma = {}".format(gamma_scale))
        else: 
            file.write("\nJust sketch (nope)")
            # raise ValueError("Invalid kernel_method.[logger info]")

    
    file.write("\nTrain A shape: {}, Train B shape: {}, label shape: {}".format(X_train1_shape, X_train2_shape, Y_train_shape))
    file.write("\nTest data shape: ({}, {}), label shape: {}".format(X_test1_shape[0], X_test1_shape[1]+X_test2_shape[1], Y_test_shape))

    file.write("\n =================== # Training info # =================== ")
    file.write("\nbatch size: {}".format(objectmodel.batch_size))   # batch大小
    file.write("\nalpha: {}".format(objectmodel.alpha)) # 学习率
    file.write("\neps: {}".format(objectmodel.eps)) # 停止误差
    file.write("\nlambda: {}".format(objectmodel.lambda_para))  # 正则化系数
    file.write("\nmax_iter: {}".format(objectmodel.max_iter))   # 最大迭代次数
    file.write("\nWAN_bandwidth: {} Mbps".format(objectmodel.WAN_bandwidth))    # 带宽
    file.write("\nmem_occupancy: {} Byte".format(objectmodel.mem_occupancy))    # 每个变量的空间占用
    file.write("\nsigmoid func: {} approximation".format(sigmoid_func))     # sigmoid 近似方法
    file.write("\nconverge_on_decide: {}".format(converge_ondecide))     # converge_是否将loss作为停止准则
    if raw_or_sketch == "raw": file.write("\nscalering_raw: {}".format(scalering_raw)) # raw数据预处理
    abspath = os.path.abspath(__file__)
    file.write("\nPython work_Path : {}".format(abspath))
    file.write("\n =================== #   Info End   # =================== \n\n")
    
    # file.close()
    # print("batch size: ", objectmodel.batch_size)
    # print("alpha: ", objectmodel.alpha)
    # print("max_iter: ", objectmodel.max_iter)
    # print("WAN_bandwidth: ", objectmodel.WAN_bandwidth)
    # print("mem_occupancy: ", objectmodel.mem_occupancy)
    # print("data source: " + flag)

def logger_test_model(objectmodel):
    file = open(objectmodel.logname, mode='a+') #  写入记录
    file.write("\n# ================== #  Test Model  # ================== #")
    file.write("\nscore: {}".format(objectmodel.score))
    file.write("\nlen(y): {}".format(objectmodel.total_num))
    file.write("\n\nPredict precision: {}\n".format(objectmodel.accuracy))

    file.write("\n# ================== #   Inference Time   # ================== #")
    # file.write("\nSecureMLModel inference_time account: {}s".format(SecureMLModel.inference_time_account))
    file.write("\nSecureMLModel inference_time account: {}s".format(objectmodel.inference_time_account))
    file.close()


def parse_input_parameter():
    """ 
    Desc
    -----
    Initialization parser.

    Tips
    ----- 
    parser.add_argument: `required`
    should be like `required=True` for almost all parameters here for model init.
    Here set to be False only for the sake of convenience while testing codes.
    """

    parser = argparse.ArgumentParser(description="Parse input parameter to initialize model and dataset.")

    parser.add_argument('-d', '--dataset-name', dest='dataset_name', required=False, type=str, metavar='STRING', help='dataset name')
    parser.add_argument('-p', '--portion', dest='portion', required=False, type=str, choices=['19', '28', '37', '46', '55'], metavar='STRING', help='data division proportion -data-')
    parser.add_argument('-m', '--modeling-method', dest='raw_or_sketch', required=False, type=str, choices=['raw', 'sketch'], metavar='STRING', help='modeling method(decided by input data type) -data-')
    parser.add_argument('-a', '--kernel', dest='kernel_method', required=False, type=str, choices=['pminhash', '0bitcws', 'rff', 'poly'], metavar='STRING', help='kernel approximation method -data-')
    parser.add_argument('-k', '--sampling-k', dest='sampling_k', required=False, type=str, metavar='STRING', help='value `k` for sampling when do kernel approximation.')
    parser.add_argument('-c', '--countsketch', dest='countsketch_', required=False, default = 0, type=int, metavar='INTEGER', help='value c for countsketch method, eg. 2, 4 ...')
    parser.add_argument('-b', '--bbitmwhash', dest='bbitmwhash_', required=False, default = 0, type=int, metavar='INTEGER', help='value b for bbitmwhash method, eg. 1, 2 ...')
    parser.add_argument('-o', '--ovr', dest='ovr', required=False, type=str, choices=['bin', 'ovr'], metavar='STRING', help='training strategy: binary or one vs rest(ovr) classification')
    parser.add_argument('-r', '--scalering-raw', dest='scalering_raw', required=False, type=str, choices=['mm', 'ss', 'na', 'ma', 'rs', 'null'], metavar='STRING', help='scalering strategy for raw data, `null` means doing nothing')
    parser.add_argument('-l', '--converge-ondecide', dest='converge_ondecide', required=False, type=str, choices=['on', 'off'], metavar='STRING', help='stopping strategy, `on`-include the loss stopping strategy, `off` otherwise')
    parser.add_argument('-s', '--sigmoid-func', dest='sigmoid_func', required=False, type=str, choices=['linear', 'cube', 'segmentation', 'original'], metavar='STRING', help='sigmoid approximation or original sigmoid')
    parser.add_argument('-al', '--alpha', dest='alpha', required=False, type=float, default = 0.001, metavar='FLOAT', help='learning rate')
    parser.add_argument('-lm', '--lambda-para', dest='lambda_para', required=False, type=float, default = 1.0, metavar='FLOAT', help='lambda parameter for penalty on loss function')
    parser.add_argument('-i', '--max-iter', dest='max_iter', required=False, type=int, metavar='INT', help='max iteraion')
    parser.add_argument('-t', '--batch-size', dest='batch_size', required=False, type=int, default = 20, metavar='INT', help='batch size')
    # parser.add_argument('-e', '--epochlist', dest='EPOCH_list', nargs='*', required=False, type=int, metavar='INT', help='Epoch list for log, zero or more parameters')
    parser.add_argument('-e', '--epochlistmax', dest='Epoch_list_max', required=False, type=int, metavar='INT', help='max epoch num for recordings')
    parser.add_argument('-f', '--logfile-write', dest='Writing_to_Final_Logfile', required=False, default = False, action="store_true", help='write to Final logfile or not, add `-f` is to write, or is not to write')
    args = parser.parse_args()

    return args

    ###

    # -d kits -p 37 -m sketch -a pminhash -k 1024 -c 4 -b 2 -o bin -r mm -l off -s linear -al 0.01 -lm 1 -i 40 -t 20 -e 1 5 10 15 20 25 30 35 40 -f
    """
    mm = MinMaxScaler()
    ss = StandardScaler()
    na = Normalizer()
    ma = MaxAbsScaler()
    rs = RobustScaler() 
    """

if __name__ == "__main__":

    """
    Use following list to save time and model weights while having trained particular epoches.
    In order to get model training output with different max_epoches, with running model training once.
    
    Tips: max_iter parameter setup should be larger than the EPOCH_list 's last number, 
    otherwise, the last number result will be None and may cause error. —— 检查最后一个元素和max_iter, 保证 list_last_num <= max_iter.

    TODO: make the list as a command line parameter with type of python list.
    """
    # EPOCH_list = [1, 5, 10, 15, 20, 25, 30, 35, 40] # Epoch_list_max=41
    Epoch_list_max = 21 # 201
    # for epoch_num in range(0, Epoch_list_max): # 前包含后不包含, 后面取的数大于最终需要的数就行
    #     if epoch_num % 5 == 0:
    #         EPOCH_list.append(epoch_num)
    # EPOCH_list[0] = 1


    ########## 读取数据 ##########
    # dataset_name = "kits"
    # portion = "37" # 19 / 28 / 37 / 46 / 55
    # raw_or_sketch = "sketch" # "raw" / "sketch"
    # kernel_method = "pminhash" # pminhash / 0bitcws / rff / poly
    # sampling_k = "1024"
    # countsketch_ = 4 # using countsketch and c = 4 / c = 8 ; not using it: c = 0
    # bbitmwhash_ = 2

    # ovr = "bin" # bin 二分类 / ovr 多分类
    # scalering_raw = "mm"    # mm / ss / na / ma / rs / null
    # converge_ondecide = "off"    # on / off # 收敛停止准则

    dataset_name = "cifar10"    # cifar10 / DailySports / webspam
    portion = "37" # 19 / 28 / 37 / 46 / 55
    raw_or_sketch = "sketch" # "raw" / "sketch"
    kernel_method = "pminhash" # 0bitcws / rff / poly
    sampling_k = "512"
    countsketch_ = 2
    bbitmwhash_ = 2 # 这里是b值, 读取文件还需要 2^b

    ovr = "ovr" # bin 二分类 / ovr 多分类
    scalering_raw = "mm"  # mm / ss / na / ma / rs / null
    converge_ondecide = "off"    # on / off


    ############ common parameters
    alpha = 0.001
    max_iter = 20 # 200
    lambda_para = 0
    batch_size = 20
    ######### other info ###########
    sigmoid_func = "linear" # linear approximation 一次函数 / cube approximation 三次函数 / segmentation approximation 分段函数 / original sigmoid 原激活函数
    #########
    Writing_to_Final_Logfile = False # 是否将本次实验结果写入此数据集的结果汇总中



    """ parameter from scripts """
    args = parse_input_parameter()

    if args.dataset_name:
        dataset_name = args.dataset_name
        portion = args.portion # 19 / 28 / 37 / 46 / 55
        raw_or_sketch = args.raw_or_sketch # "raw" / "sketch"
        kernel_method = args.kernel_method # 0bitcws / rff / poly
        sampling_k = args.sampling_k
        countsketch_ = args.countsketch_
        bbitmwhash_ = args.bbitmwhash_

        ovr = args.ovr # bin 二分类 / ovr 多分类
        scalering_raw = args.scalering_raw  # mm / ss / na / ma / rs / null
        converge_ondecide = args.converge_ondecide # on / off

        alpha = args.alpha # 学习率
        max_iter = args.max_iter
        lambda_para = args.lambda_para
        batch_size = args.batch_size

        # EPOCH_list = args.EPOCH_list
        Epoch_list_max = args.Epoch_list_max

        ######### other info ###########
        sigmoid_func = args.sigmoid_func # linear approximation 一次函数 / cube approximation 三次函数 / segmentation approximation 分段函数 / original sigmoid 原激活函数
        #########
        Writing_to_Final_Logfile = args.Writing_to_Final_Logfile # 是否将本次实验结果写入此数据集的结果汇总中

        print("[Attention] args come from shell, not default ones.")
    else:
        print("[Attention] args are default ones, not from shell.")

    # 基础测试
    # X_data, y_data, X_test, y_test = read_data()
    # X_data, y_data, X_test, y_test = read_sampling_data()
    # 理想的集中数据集
    # X_data, y_data, X_test, y_test = read_encoded_data()
    # print(X_data.shape, X_data.shape[0], X_data.shape[1], y_data.shape, X_test.shape, y_test.shape)

    # 纵向划分的数据集
    # X_train1, X_train2, Y_train, X_test1, X_test2, Y_test = read_distributed_encoded_data()

    # Raw data
    # X_train1, X_train2, Y_train, X_test1, X_test2, Y_test = read_distributed_data()

    # Sketch data
    # data loader
    X_train1, X_train2, Y_train, X_test1, X_test2, Y_test, gamma_scale = read_distributed_data_raw_or_sketch(dataset_name, raw_or_sketch, 
                                                    kernel_method, portion, sampling_k, ovr, countsketch_, bbitmwhash_, scalering_raw)
    

    if raw_or_sketch == "sketch" and kernel_method in ["rff", "poly"] and gamma_scale == -1:
        raise ValueError("gamma_scale not updated.")
    
    # print(X_train1.shape, X_train2.shape, X_train1.shape[1], X_train2.shape[1], Y_train.shape, X_test1.shape, Y_test.shape)

    ########## 权重初始化 ##########
    np.random.seed(100)
    # # 正态
    # weight_vector = np.random.normal(0.0, 0.0001, X_data.shape[1]).reshape(1, -1)
    # # 全1
    # weight_vector = np.ones(X_data.shape[1]).reshape(1,-1)
    # # 随机
    # weight_vector = np.random.random(X_data.shape[1]).reshape(1, -1)
    # weight = (0.00001-0.0) * weight_vector + 0.0
    # # 全0
    # weight_vector = np.zeros(X_data.shape[1]).reshape(1, -1)

    # 纵向划分分布式
    weight_vector = np.zeros(X_train1.shape[1]+X_train2.shape[1]).reshape(1, -1)
    # print(weight_vector)


    ########## 模型实例化 ##########
    # 伪纵向划分——检验算法正确性
    # LogisticRegressionModel = LogisticRegression(weight_vector = weight_vector, batch_size = 20, 
    #                 max_iter = 1000, alpha = 0.0001, eps = 1e-6, ratio = 0.7, data_tag = 'sparse')  # 0.956140350877193
                    # splice 分布式: Predict precision:  0.8496551724137931
                    # splice 分布式 0.9062068965517242
    # 理想集中
    # LogisticRegressionModel = LogisticRegression(weight_vector = weight_vector, batch_size = 20, 
    #                 max_iter = 2000, alpha = 0.0001, eps = 1e-6, penalty = None, lambda_para = 1, data_tag=None)  
                    # breast_cancer: 0.9649122807017544
                    # splice: 0.8482758620689655
                    # splice 集中 0.9062068965517242
    # 纵向划分分布式
    SecureMLModel = SecureML(weight_vector = weight_vector, batch_size = batch_size, 
                    max_iter = max_iter, alpha = alpha, eps = 1e-5, ratio = 0.7, penalty = None, lambda_para = lambda_para, 
                    data_tag = None, ovr = ovr,
                    sketch_tag = raw_or_sketch, countsketch_c = countsketch_, bbitmwhash_b = bbitmwhash_, 
                    dataset_name = dataset_name, kernel_method = kernel_method, sampling_k = sampling_k,
                    Epoch_list_max = Epoch_list_max)
                    # splice 分布式 0.9062068965517242
    # LogisticRegressionModel = LogisticRegression(weight_vector = weight_vector, batch_size = 20, 
    #                 max_iter = 600, alpha = 0.0001, eps = 1e-6, ratio = 0.7, penalty = None, lambda_para = 1, data_tag = None)

    # 两部分数据集
    # sparse: 12.54981803894043 s,       Predict precision:  0.9062068965517242    Iteration 645
    # non-sparse: 30.390305995941162 s,  Predict precision:  0.9062068965517242    Iteration 645

    # 集中：
    # sparse:  14.546779870986938 s   Predict precision:  0.9062068965517242

    logger_info(SecureMLModel, dataset_name, raw_or_sketch, kernel_method, portion, sampling_k, countsketch_, bbitmwhash_, gamma_scale,
                X_train1.shape, X_train2.shape, X_test1.shape, X_test2.shape, Y_train.shape, Y_test.shape, sigmoid_func, scalering_raw, converge_ondecide)

    ########## 训练 ##########
    import time
 
    time_start = time.time()
    
    # 理想集中和伪分布式
    # LogisticRegressionModel.fit_model(X_data, y_data, X_data.shape[0])

    # 纵向划分分布式
    # indice_littleside = X_train1.shape[1]
    # LogisticRegressionModel.fit_model_distributed_input(X_train1, X_train2, Y_train, X_train1.shape[0], indice_littleside)

    # 纵向分布保护隐私的分布式
    indice_littleside = X_train1.shape[1]

    if SecureMLModel.ovr == "bin":
        SecureMLModel.Binary_Secure_Classifier(X_train1, X_train2, Y_train, X_train1.shape[0], (X_train1.shape[1]+X_train2.shape[1]), indice_littleside, converge_ondecide)
        # SecureMLModel.fit_model_secure_2process(X_train1, X_train2, Y_train, X_train1.shape[0], indice_littleside)
    elif SecureMLModel.ovr == "ovr":
        SecureMLModel.OneVsRest_Secure_Classifier(X_train1, X_train2, X_test1, X_test2, Y_train, Y_test, converge_ondecide)
    
    # time_end = time.time()
    # print("SecureMLModel comm_time account: ", SecureMLModel.train_time_account)
    # print('Total time cost: ' + str(time_end - time_start + SecureMLModel.train_time_account) + 's')

    
    # file = open(SecureMLModel.logname, mode='a+') #  写入记录
    # file.write("\n# ================== #   Train Time   # ================== #")
    # file.write("\nSecureMLModel comm_time account: {}s".format(SecureMLModel.train_time_account))
    # # 总开销(online和offline的计算和通信时间):
    # file.write("\nTotal time cost: {}s".format(time_end - time_start))

    # plt.plot(LogisticRegressionModel.loss_history)
    # plt.show()

    ########## 测试 ##########
    # 理想集中和伪分布式
    # LogisticRegressionModel.predict(X_test, y_test)
    # 纵向划分分布式
    if SecureMLModel.ovr == "bin":
        if SecureMLModel.EPOCH_list is not None:
            SecureMLModel.predict_forEpochs_Bin(X_test1, X_test2, Y_test)
        else:
            SecureMLModel.predict_distributed(X_test1, X_test2, Y_test)
            logger_test_model(SecureMLModel)
    elif SecureMLModel.ovr == "ovr":
        # logger_test_model(SecureMLModel)
        pass

    time_end = time.time()

    file = open(SecureMLModel.logname, mode='a+') #  写入记录
    # file.write("\n# ================== #   Train Time   # ================== #")
    # file.write("\nSecureMLModel comm_time account: {}s".format(SecureMLModel.train_time_account))
    # # 总开销(online和offline的计算和通信时间):
    file.write("\n\n\nTotal time cost of the exhausting running: {}s (≈ {}h)".format(time_end - time_start, (time_end - time_start) / 3600))
    file.close()

    

    