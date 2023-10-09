import os
PATH_DATA = '/home/user/zbz/SSHE/data/'
PATH_FILE = '/home/user/zbz/SSHE/logistic_regression/nompc/result/'

dataset_name = "DailySports"  # cifar10
portion = "37"  # 19 / 28 / 37 / 46 / 55
raw_or_sketch = "sketch"  # "raw" / "sketch"
kernel_method = "poly"  # pminhash / 0bitcws / rff / poly
sampling_k = "2048"
countsketch = 2
bbitmwhash = 2  # 这里是b值, 读取文件还需要 2^b
# scalering_raw = "mm"  # 原数据集归一化方法 mm / ss / na / ma / rs / null
# converge_ondecide = "off"  # on / off
# multi_class = 'ovr'  # 二分类bin还是多分类ovr

# 日志文件存储
if raw_or_sketch == "sketch":
    if kernel_method == "pminhash":
        logname = dataset_name + "_" + portion + "_" + kernel_method + sampling_k + "_" + str(countsketch)
    elif kernel_method == "0bitcws":
        logname = dataset_name + "_" + portion + "_" + kernel_method + sampling_k + "_" + str(2 ** bbitmwhash)
    else:
        # rff, poly
        logname = dataset_name + "_" + portion + "_" + kernel_method + sampling_k
else:
    logname = dataset_name + "_raw"
log_path = os.path.join(PATH_FILE, logname)
if not os.path.exists(log_path):
    os.makedirs(log_path)

# batch_size_lst = [2, 4, 8, 16, 32]
# alpha_lst = [0.0001, 0.001, 0.01, 0.1, 1]
# lambda_para_lst = [0, 0.001, 0.01, 0.1, 1]
# # C_lst = [0.001, 0.01, 0.1, 1, 10, 100]
# C_lst = [1]
# sigmoid_func_lst = ["linear", "cube", "segmentation", "origin"]
batch_size_lst = [2]
alpha_lst = [0.0001]
lambda_para_lst = [0, 0.001, 0.01, 0.1, 1]
C_lst = [1]
sigmoid_func_lst = ["linear"]

max_iter = 200

acc_dict = dict()

for sigmoid_func in sigmoid_func_lst:
    for batch_size in batch_size_lst:
        for alpha in alpha_lst:
            for lambda_para in lambda_para_lst:
                for C in C_lst:
                    file_name = os.path.join(log_path,
                                             "alpha={}_batch_size={}_max_iter={}_lambda_para={}_C={}_sigmoid_func={}.txt".format(
                                                 alpha, batch_size, max_iter, lambda_para, C, sigmoid_func))
                    string_lst = ["alpha=", "batch_size=", "max_iter=", "lambda_para=", "C=", "sigmoid_func=", "best_acc=", "best_epoch="]
                    value_lst = [None, None, None, None, None, None, None, None]
                    # string1 = "alpha="
                    # string2 = "batch_size="
                    # string3 = "max_iter="
                    # string4 = "lambda_para="
                    # string5 = "C="
                    # string6 = "sigmoid_func="
                    # string7 = "best_acc="
                    # string8 = "best_epoch="

                    with open(file_name, "r") as f:
                        # 对集成在一起的文件这么做
                        # for line in f:
                        #     for i, string in enumerate(string_lst):
                        #         if string in line:
                        #             index = line.index(string) + len(string)
                        #             # print(index)
                        #             if index < len(line):
                        #                 value_lst[i] = line[index:line.index("\n")]
                        # 对单个结果存储单个文件这么做
                        for line in f:
                            if "best_acc=" in line:
                                index = line.index("best_acc=") + len("best_acc=")
                                # print(index)
                                if index < len(line):
                                    best_acc = line[index:line.index("\n")]

                            if "best_epoch=" in line:
                                index = line.index("best_epoch=") + len("best_epoch=")
                                # print(index)
                                if index < len(line):
                                    best_epoch = line[index:line.index("\n")]
                    acc_dict.update({best_acc: [best_epoch, sigmoid_func, batch_size, alpha, lambda_para, C]})

key_lst = list(acc_dict.keys())
key_lst=list(map(float, key_lst))
print(key_lst, type(key_lst[0]))

# for acc_tuple in acc_lst1:
#     print(acc_tuple)