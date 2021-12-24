import os
from glob import glob
import cv2
import numpy as np
import pandas as pd
import pickle
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from main import *


def get_image_paths0(data_path, categories):
    train_image_paths = []
    test_image_paths = []

    train_labels = []
    test_labels = []
    print("获得训练集数据路径")
    for category in categories:
        image_paths = glob(os.path.join(data_path, 'train', category, '*.png'))
        path = os.path.join(data_path, 'train', category)
        files = os.listdir(path)

        for i in range(len(files)):
            train_image_paths.append(image_paths[i])
            train_labels.append(category)
        print(path, '读取完成！', '->', len(files))
    print("获得测试集数据路径")
    image_paths = glob(os.path.join(data_path, 'test', '*.png'))
    path = os.path.join(data_path, 'test')
    files = os.listdir(path)
    for i in range(len(files)):
        test_image_paths.append(image_paths[i])
    print(path, '读取完成！', '->', len(files))
    return train_image_paths, test_image_paths, train_labels
def get_train_feat0(image_paths):
    train = []
    for image_path in image_paths:
        img = cv2.imread(image_path)
        img = segment_plant(img)
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        r, g, b = cv2.split(img)
        r_mean = np.mean(r)
        g_mean = np.mean(g)
        b_mean = np.mean(b)
        r_std = np.std(r)
        g_std = np.std(g)
        b_std = np.std(b)
        r_offset = (np.mean(np.abs((r - r_mean) ** 3))) ** (1. / 3)
        g_offset = (np.mean(np.abs((g - g_mean) ** 3))) ** (1. / 3)
        b_offset = (np.mean(np.abs((b - b_mean) ** 3))) ** (1. / 3)

        # # img= cv2.medianBlur(img, 3)  # 中值滤波
        # one = np.cumsum(cv2.calcHist([img], [1], None, [256], [0, 255],accumulate=True))
        # # one = np.std(one).tolist()
        # one = (one/(img.shape[0]*img.shape[1])).tolist()
        one = np.log1p([r_mean, g_mean, b_mean, r_std, g_std, b_std, r_offset, g_offset, b_offset]).tolist()
        train.append(one)
    return train
if __name__ == '__main__':
    data_path = "."
    categories = ['Black-grass', 'Charlock', 'Cleavers', 'Common Chickweed', 'Common wheat', 'Fat Hen',
                  'Loose Silky-bent',
                  'Maize', 'Scentless Mayweed', 'Shepherds Purse', 'Small-flowered Cranesbill', 'Sugar beet']
    CATE2ID = {v: k for k, v in enumerate(categories)}
    train_image_paths, test_image_paths, train_labels = get_image_paths0(data_path, categories)
    trian_image_labels_id = [CATE2ID[x] for x in train_labels]
    # train_image_paths, trian_image_labels_id= shuffle(train_image_paths, trian_image_labels_id, random_state=0)
    print("开始提取训练集特征")
    if not os.path.exists('train_X_color.pkl'):
        train_X = get_train_feat0(train_image_paths)
        pickle.dump(train_X, open('train_X_color.pkl', 'wb'))
    else:
        train_X = pickle.load(open('train_X_color.pkl', 'rb'))
    print("提取完成")
    train_y = trian_image_labels_id
    train_X, train_y = shuffle(train_X, train_y, random_state=0)
    print("开始训练")
    train_X, x_test, train_y, y_test = train_test_split(train_X, train_y, test_size=0.2)
    print("训练结束")
    print(len(train_X[0]))
    model = svm.SVC(kernel='linear', probability=True, gamma='auto', C=50).fit(train_X, train_y)
    print("开始提取测试集特征")
    if not os.path.exists('test_X_color.pkl'):
        test_X = get_train_feat0(test_image_paths)
        pickle.dump(test_X, open('test_X_color.pkl', 'wb'))
    else:
        test_X = pickle.load(open('test_X_color.pkl', 'rb'))
    print("提取完成")
    print("训练集准确率=====>>>>>>", model.score(train_X, train_y))
    print("验证集准确率=====>>>>>>", model.score(x_test, y_test))
    preds = model.predict(test_X)
    print("预测结束===========================>")
    # print(preds)

    test = []
    for i in range(preds.shape[0]):
        test.append(categories[preds[i]])
    print(len(test))
    sample = pd.read_csv("sample_submission.csv")
    submission = pd.DataFrame({'file': sample['file'], 'species': test})
    submission.to_csv('color+SVM.csv', index=False)