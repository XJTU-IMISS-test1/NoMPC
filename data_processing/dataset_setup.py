import os

dataset_folder = '/home/user/zbz/SSHE/data'

def setup_dataset_folder(path):
    """
    For particular dataset, generate dataset folders, 
    including portion/ sketch/ countsketch ...
    """
    folder = os.path.exists(path)
    if not folder:                   #判断是否存在文件夹如果不存在则创建为文件夹
        os.makedirs(path)            #makedirs 创建文件时如果路径不存在会创建这个路径
        print("---  new folder...  ---")
    '''
    新建数据集目录结构, 处理数据: sketch-countsketch
    '''