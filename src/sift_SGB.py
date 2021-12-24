import os
from glob import glob
import cv2
import numpy as np
import pandas as pd
import pickle
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import SGDClassifier
from sklearn.multiclass import OneVsOneClassifier
from main import segment_plant

def get_image_paths(data_path, categories):
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


def build_vocabulary(image_paths, k, length):
    bow_kmeans_trainer = cv2.BOWKMeansTrainer(k)
    sift = cv2.SIFT_create()
    for i in range(length):
        for j in [0, 263, 653, 940, 1551, 1772, 2247, 2901, 3122, 3638, 3869, 4365]:
            img = cv2.imread(image_paths[j + i])
            gray = segment_plant(img)
            kp = sift.detect(gray, None)
            if len(kp) != 0:
                bow_kmeans_trainer.add(sift.compute(gray, sift.detect(gray, None))[1])
    vocab = bow_kmeans_trainer.cluster()
    return vocab


def get_train_feat(image_paths, vocab, k):
    flann_params = dict(algorithm=1, tree=5)
    flann = cv2.FlannBasedMatcher(flann_params, {})
    train = []
    sift = cv2.SIFT_create()
    bow_img_descriptor_extractor = cv2.BOWImgDescriptorExtractor(sift, flann)
    bow_img_descriptor_extractor.setVocabulary(vocab)
    for image_path in image_paths:
        img = cv2.imread(image_path)
        gray = segment_plant(img)
        one = bow_img_descriptor_extractor.compute(gray, sift.detect(gray, None))
        if one is None:
            train.append(np.array([[0 for i in range(k)]]))
        else:
            train.append(one)
    return train


if __name__ == '__main__':
    data_path = "."
    categories = ['Black-grass', 'Charlock', 'Cleavers', 'Common Chickweed', 'Common wheat', 'Fat Hen',
                  'Loose Silky-bent',
                  'Maize', 'Scentless Mayweed', 'Shepherds Purse', 'Small-flowered Cranesbill', 'Sugar beet']
    CATE2ID = {v: k for k, v in enumerate(categories)}
    train_image_paths, test_image_paths, train_labels = get_image_paths(data_path, categories)
    trian_image_labels_id = [CATE2ID[x] for x in train_labels]
    print("开始建立词汇表===========================>")
    if not os.path.exists('vocab.pkl'):
        vacob = build_vocabulary(train_image_paths, 100, 200)
        pickle.dump(vacob, open('vocab.pkl', 'wb'))
    else:
        vacob = pickle.load(open('vocab.pkl', 'rb'))
    print(vacob.shape)
    print("词汇表建立完成===========================>")
    # 打乱
    # train_image_paths, trian_image_labels_id= shuffle(train_image_paths, trian_image_labels_id, random_state=0)
    print("开始提取训练集BOW描述===========================>")
    if not os.path.exists('train_X.pkl'):
        train_X = get_train_feat(train_image_paths, vacob, 100)
        pickle.dump(train_X, open('train_X.pkl', 'wb'))
    else:
        train_X = pickle.load(open('train_X.pkl', 'rb'))
    # train_X = get_train_feat(train_image_paths, vacob, 50)
    # print(train_X)
    # exit(0)
    train = []
    # print()
    for train1 in train_X:
        # print(train1)
        train.append(train1.tolist()[0])
    # train = np.array(train)
    # print(train)
    print("提取完成===========================>")
    train_y = trian_image_labels_id
    # train_y = np.array(train_y)
    print("开始提取测试集BOW描述===========================>")
    if not os.path.exists('test_X.pkl'):
        test_X = get_train_feat(test_image_paths, vacob, 100)
        pickle.dump(test_X, open('test_X.pkl', 'wb'))
    else:
        test_X = pickle.load(open('test_X.pkl', 'rb'))
    #test_X = get_train_feat(test_image_paths, vacob, 50)
    test = []
    for test1 in test_X:
        test.append(test1.tolist()[0])
    # test = np.array(test)
    print("提取完成===========================>")
    print("开始训练===========================>")


    # 打乱顺序
    train, train_y = shuffle(train, train_y)
    # 划分
    train, x_test, train_y, y_test = train_test_split(train,train_y,test_size=0.2)
    # 0.70277 /C=100 0.70340 / C =50/64/55 0.70969 C=81 0.70654  C =72  0.70843
    model = svm.SVC(kernel='linear', probability=True, gamma='auto', C=20).fit(train, train_y)
    # 0.6402 model = KNeighborsClassifier(n_neighbors=5).fit(train,train_y)
    # model = make_pipeline(StandardScaler(),
    #          SGDClassifier(max_iter=1000, tol=1e-3)).fit(train,train_y)
    print("训练结束===========================>")
    print("训练集准确率=====>>>>>>", model.score(train, train_y))
    print("验证集准确率=====>>>>>>", model.score(x_test, y_test))
    print("开始预测===========================>")
    preds = model.predict(test)
    print("预测结束===========================>")
    # print(preds)
    test = []
    for i in range(preds.shape[0]):
        test.append(categories[preds[i]])
    print(len(test))
    sample = pd.read_csv("sample_submission.csv")
    submission = pd.DataFrame({'file': sample['file'], 'species': test})
    submission.to_csv('sift+SVM.csv', index=False)