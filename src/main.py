# coding=utf-8
import numpy as np
import cv2
import pandas as pd
import os
import matplotlib
import matplotlib.pyplot as plt
from glob import glob
import pickle
import seaborn as sns
import sklearn
from skimage.feature import local_binary_pattern
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LassoCV,RidgeCV
from sklearn.model_selection import KFold, cross_val_score
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier,OneVsOneClassifier
from sklearn.model_selection import train_test_split
from sklearn import svm
import random
from sklearn.linear_model import LogisticRegression,SGDClassifier
import joblib
import warnings

# 忽略警告
warnings.filterwarnings("ignore")


def load_data():
    '''
    读取训练集和测试集
    {0: 'Black-grass', 1: 'Charlock', 2: 'Cleavers', 3: 'Common Chickweed', 4: 'Common wheat', 5: 'Fat Hen', 6: 'Loose Silky-bent',
    7: 'Maize', 8: 'Scentless Mayweed', 9: 'Shepherds Purse', 10: 'Small-flowered Cranesbill', 11: 'Sugar beet'}
    '''
    #整个训练集plant-seedlings-classification文件夹的路径
    DATA_FOLDER = "."
    # train_folder保存为train文件夹的路径
    TRAIN_FOLDER = os.path.join(DATA_FOLDER,'train')
    # test_folder保存为test文件夹的路径
    TEST_FOLDER = os.path.join(DATA_FOLDER,'test')
    print(os.listdir(TRAIN_FOLDER))
    print(os.listdir(TEST_FOLDER)[:10])

    # 读取训练集
    train = {}
    image_type={}
    i=0
    #train{}为一个字典  train.key()为plant的标签 对应的train[label]为所有的训练的图片的numpy矩阵
    for plant_name in os.listdir(TRAIN_FOLDER):
        plant_path = os.path.join(TRAIN_FOLDER, plant_name)
        label = plant_name
        train[i] = []
        for image_path in glob(os.path.join(plant_path,'*png')):
            image = cv2.imread(image_path)
            train[i].append(image)
            # 旋转 90 180 270 加入训练集
            # image=np.rot90(image)
            # train[i].append(image) 
            # image=np.rot90(image)
            # train[i].append(image) 
            # image=np.rot90(image)
            # train[i].append(image) 
        print(plant_path,'读取完成！',label,'->',len(train[i]))
        image_type[i]=label
        i+=1
    print(image_type)

    # 读取测试集
    test_data=[]
    for image_path in glob(os.path.join(TEST_FOLDER,'*png')):
        image = cv2.imread(image_path)
        test_data.append(image)
    print('测试集长度：',len(test_data))
    print("读取完成！")
    return train,test_data,image_type

#显示一个label的前6张图片
def plot_for_class(label):
    nb_rows = 2
    nb_cols = 3
    fig, axs = plt.subplots(nb_rows, nb_cols, figsize=(5, 5))
    n = 0
    for i in range(0, nb_rows):
        for j in range(0, nb_cols):
            axs[i, j].xaxis.set_ticklabels([])
            axs[i, j].yaxis.set_ticklabels([])
            axs[i, j].imshow(train[label][n])
            n += 1

def create_mask_for_plant(image):
    #bgr转化为hsv
    image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    sensitivity = 35
    lower_hsv = np.array([60 - sensitivity, 100, 50])
    upper_hsv = np.array([60 + sensitivity, 255, 255])
    # lower_hsv = np.array([25, 40, 40])
    # upper_hsv = np.array([80, 255, 255])
    
    #二值化
    mask = cv2.inRange(image_hsv, lower_hsv, upper_hsv)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    
    # 形态学开操作 先腐蚀后膨胀
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    
    return mask

def segment_plant(image):
    mask = create_mask_for_plant(image)
    # 利用掩膜进行图像混合
    # 求交集 、 掩膜提取图像
    output = cv2.bitwise_and(image, image, mask = mask)
    return output

def sharpen_image(image):
    # 高斯模糊 线性平滑
    image_blurred = cv2.GaussianBlur(image, (0, 0), 3)
    # 以一定的权重进行图像融合
    image_sharp = cv2.addWeighted(image, 1.5, image_blurred, -0.5, 0)
    return image_sharp

def calculate_largest_contour_area(contours):
    if len(contours) == 0:
        return 0
    c = max(contours, key=cv2.contourArea)
    return cv2.contourArea(c)

def calculate_contours_area(contours, min_contour_area = 250):
    area = 0
    for c in contours:
        c_area = cv2.contourArea(c)
        if c_area >= min_contour_area:
            area += c_area
    return area

def get_train_(train):
    '''
        获得 训练集的形状特征
    '''
    areas = []
    larges_contour_areas = []
    labels = []
    nb_of_contours = []
    images_height = []
    images_width = []
    for class_label in train.keys():
        for image in train[class_label]:
            mask = create_mask_for_plant(image)
            # mask = segment_plant(mask)
            contours = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
            area = calculate_contours_area(contours)
            largest_area = calculate_largest_contour_area(contours)
            height, width, channels = image.shape
            images_height.append(height)
            images_width.append(width)
            areas.append(area)
            nb_of_contours.append(len(contours))
            larges_contour_areas.append(largest_area)
            labels.append(class_label)
    features_df = pd.DataFrame()
    features_df["label"] = labels
    features_df["area"] = areas
    features_df["largest_area"] = larges_contour_areas
    features_df["number_of_components"] = nb_of_contours
    # features_df["height"] = images_height
    # features_df["width"] = images_width
    return features_df

def get_test_(test_data):
    '''
        获得 测试集的形状特征
    '''
    test_areas = []
    test_larges_contour_areas = []
    test_nb_of_contours = []
    test_images_height = []
    test_images_width = []
    for image in test_data:
        mask = create_mask_for_plant(image)
        contours = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
        area = calculate_contours_area(contours)
        largest_area = calculate_largest_contour_area(contours)
        height, width, channels = image.shape
        test_images_height.append(height)
        test_images_width.append(width)
        test_areas.append(area)
        test_nb_of_contours.append(len(contours))
        test_larges_contour_areas.append(largest_area)
    features_test = pd.DataFrame()
    features_test["area"] = test_areas
    features_test["largest_area"] = test_larges_contour_areas
    features_test["number_of_components"] = test_nb_of_contours
    # features_test["height"] = test_images_height
    # features_test["width"] = test_images_width
    return features_test


if __name__=='__main__':
    # 将数据dump保存  下次不需要重新读取
    # train_data,test_data,image_type=load_data()
    # pickle.dump(train_data,open("train_data.pkl",'wb'))
    # pickle.dump(test_data,open("test_data.pkl",'wb'))
    # pickle.dump(image_type,open("image_type.pkl",'wb'))


    train_data=pickle.load(open("train_data.pkl",'rb'))
    test_data=pickle.load(open("test_data.pkl",'rb'))
    image_type=pickle.load(open("image_type.pkl",'rb'))
    features_df=get_train_(train_data)
    features_test=get_test_(test_data)


    train=features_df.iloc[:,1:]
    train = train.apply(lambda x: (x - x.mean()) / (x.std())) #归一化
    label=features_df.iloc[:,:1]
    train=np.array(train)
    label=np.array(label)

    print(train.shape,label.shape)

    # 打乱顺序
    train, label = sklearn.utils.shuffle(train, label)
    # 划分测试集和训练集
    x_train, x_test, y_train, y_test = train_test_split(train,label,test_size=0.2)

    
    # model=svm.SVC(kernel='linear',gamma='auto',probability=True,C=5)
    # model = OneVsRestClassifier(svm.SVC(kernel='linear',probability=True,C=20)) #0.28
    # model = OneVsOneClassifier(svm.SVC(kernel='linear',gamma='auto',probability=True,C=1)) #0.28
    model=OneVsRestClassifier(LogisticRegression(solver="liblinear",C=10)) #0.29
    # model=OneVsOneClassifier(LogisticRegression(solver="lbfgs"))

    model.fit(x_train,y_train)
    # print( len(model.estimators_) )
    print("训练集准确率=====>>>>>>",model.score(x_train,y_train))
    print("验证集准确率=====>>>>>>",model.score(x_test,y_test))

    # joblib.dump(model,'svc.pkl')
    #重新加载model，只有保存一次后才能加载model
    # clf3=joblib.load('sklearn_save/clf.pkl')


    features_test = features_test.apply(lambda x: (x - x.mean()) / (x.std())) #归一化

    preds = model.predict(features_test)
    test=[]
    for i in range(preds.shape[0]):
        test.append(image_type[preds[i]])
    print(len(test))
    sample = pd.read_csv("sample_submission.csv")
    submission = pd.DataFrame({'file': sample['file'], 'species': test})
    submission.to_csv('HOG+SVM.csv', index=False) 
