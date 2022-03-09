# coding=utf-8
import os, random, shutil
 
def moveFile(fileDir):
    pathDir = os.listdir(fileDir)  # 取图片的原始路径
    filenumber = len(pathDir)
    picknumber = int(filenumber * ratio)  # 按照rate比例从文件夹中取一定数量图片
    sample = random.sample(pathDir, picknumber)  # 随机选取picknumber数量的样本图片
    for name in sample:
        shutil.move(os.path.join(fileDir, name), os.path.join(tarDir, name))
    return
 
if __name__ == '__main__':
    ori_path = '/hd1/cropsy/dataset' # 最开始train的文件夹路径

    split_train = False # train was splitted
    if split_train:
        split_Dir = '/hd1/cropsy/formatted_dataset/train' # 移动到新的文件夹路径

        ratio = 0.70  # 抽取比例  ******自己改*******
        for firstPath in os.listdir(ori_path):
            fileDir = os.path.join(ori_path, firstPath)  # 原图片文件夹路径
            tarDir = os.path.join(split_Dir, firstPath)  # val下子文件夹名字
            if not os.path.exists(tarDir): #如果val下没有子文件夹，就创建
                os.makedirs(tarDir)
            moveFile(fileDir)  # 从每个子类别开始逐个划分
        print("Successfully split the train!")

    split_val = False #True
    if split_val: # validation was splitted
        split_Dir = '/hd1/cropsy/formatted_dataset/val'

        ratio = 0.67  # 20% of the whole dataset # 抽取比例  ******自己改*******
        for firstPath in os.listdir(ori_path):
            fileDir = os.path.join(ori_path, firstPath)  # 原图片文件夹路径
            tarDir = os.path.join(split_Dir, firstPath)  # val下子文件夹名字
            if not os.path.exists(tarDir): # 如果val下没有子文件夹，就创建
                os.makedirs(tarDir)
            moveFile(fileDir)  # 从每个子类别开始逐个划分
        print("Successfully split the val!")

    split_test = True
    if split_test: # validation was splitted
        split_Dir = '/hd1/cropsy/formatted_dataset/test'

        ratio = 1  # 10% of the whole dataset # 抽取比例  ******自己改*******
        for firstPath in os.listdir(ori_path):
            fileDir = os.path.join(ori_path, firstPath)  # 原图片文件夹路径
            tarDir = os.path.join(split_Dir, firstPath)  # val下子文件夹名字
            if not os.path.exists(tarDir): # 如果val下没有子文件夹，就创建
                os.makedirs(tarDir)
            moveFile(fileDir)  # 从每个子类别开始逐个划分
        print("Successfully split the test!")
