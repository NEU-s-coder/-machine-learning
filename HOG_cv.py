#coding:utf-8
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pickle
import tqdm
from main import *

def get_HOG_feature(img,hog):
    # cv2.imshow('0',img)
    # 提取出绿色部分
    # img = create_mask_for_plant(img)
    img = segment_plant(img)
    # 高斯模糊+图像权值融合
    # img = sharpen_image(img)

    # cv2.imshow('1',img)
    # cv2.waitKey(0)
    img = cv2.resize(img,(64,64))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    feature=hog.compute(img, winStride=(32,32), padding=(0,0)).flatten()
    return feature

if __name__ == '__main__':
    # 将数据dump保存  下次不需要重新读取
    if not os.path.exists('train_data.pkl'):
        train_data,test_data,image_type=load_data()
        pickle.dump(train_data,open("train_data.pkl",'wb'))
        pickle.dump(test_data,open("test_data.pkl",'wb'))
        pickle.dump(image_type,open("image_type.pkl",'wb'))
    else:
        train_data=pickle.load(open("train_data.pkl",'rb'))
        test_data=pickle.load(open("test_data.pkl",'rb'))
        image_type=pickle.load(open("image_type.pkl",'rb'))

    print(train_data.keys())
    print(len(test_data))
    print(image_type)

    # cv2.namedWindow('0',cv2.WINDOW_NORMAL)


    #定义对象hog，同时输入定义的参数，剩下的默认即可
    winSize = (64,64)
    blockSize = (32,32)
    blockStride = (8,8)
    cellSize = (8,8)
    nbins = 9
    hog = cv2.HOGDescriptor( winSize,blockSize,blockStride,cellSize, nbins )


    train_feature=[]
    train_label=[]
    for class_label in train_data.keys():
        for image in train_data[class_label]:
            feature = get_HOG_feature(image,hog)
            train_feature.append(feature)
            train_label.append(class_label)
    
    print("**************************")
    print(len(train_feature))
    print(len(train_label)) 
    train_feature=np.array(train_feature)
    train_label=np.array(train_label)
    
    print(train_feature.shape)
    print(train_label.shape)

    # PCA降维
    pca = PCA(n_components=50)  # 自动选择特征个数  'mle'
    print("**************************")
    pca.fit(train_feature)
    print("**************************")
    print("降维前train.shape:{0}".format(train_feature.shape))
    train_feature = pca.transform(train_feature)
    print("降维后train.shape:{0}".format(train_feature.shape))


    # 打乱顺序
    train_feature, train_label = sklearn.utils.shuffle(train_feature, train_label)
    # 划分
    x_train, x_test, y_train, y_test = train_test_split(train_feature,train_label,test_size=0.2,random_state=0)

    # from sklearn import tree
    # clf = tree.DecisionTreeClassifier()

    # model = OneVsOneClassifier(svm.SVC(kernel='linear',probability=True,C=1))
    # model = OneVsRestClassifier(svm.SVC(kernel='linear',probability=True))
    model = svm.SVC(kernel='linear',probability=True,gamma='auto',C=0.5)
    # model=  SGDClassifier(tol=1e-3)
    # model=OneVsOneClassifier(LogisticRegression(solver="liblinear",C=1)) #0.29
    # model = XGBClassifier( objective='multi：softmax')

    
    print("开始训练====>>>")
    if not os.path.exists('HOG+SVM.pkl'):
        model.fit(x_train,y_train)
        # joblib.dump(model,'HOG+SVM.pkl')
    else:
        #重新加载model，只有保存一次后才能加载model
        model=joblib.load('HOG+SVM.pkl')
    print("训练集准确率=====>>>>>>",model.score(x_train,y_train))
    print("验证集准确率=====>>>>>>",model.score(x_test,y_test))


    test_feature=[]
    for image in test_data:
            feature = get_HOG_feature(image,hog)
            test_feature.append(feature)
    print(len(test_feature))
    test_feature = np.array(test_feature)

    print("降维前train.shape:{0}".format(test_feature.shape))
    test_feature = pca.transform(test_feature)
    print("降维后train.shape:{0}".format(test_feature.shape))

    preds = model.predict(test_feature)
    test=[]
    for i in range(preds.shape[0]):
        test.append(image_type[preds[i]])
    print(len(test))
    sample = pd.read_csv("sample_submission.csv")
    submission = pd.DataFrame({'file': sample['file'], 'species': test})
    submission.to_csv('HOG+SVM.csv', index=False)    
