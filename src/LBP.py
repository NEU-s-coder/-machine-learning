#coding:utf-8
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pickle
import tqdm
from main import *


# def Lbp(img):
#     #使用LBP方法提取图像的纹理特征.
#     lbp=local_binary_pattern(train_data[i],n_point,radius,'default');
#     #统计图像的直方图
#     max_bins = int(lbp.max() + 1);
#     #hist size:256
#     train_hist[i], _ = np.histogram(lbp, normed=True, bins=max_bins, range=(0, max_bins));
def get_LBP_feature(img):
    # 提取出绿色部分
    # img = create_mask_for_plant(img)
    img = segment_plant(img)

    img = cv2.resize(img,(128,128))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # LBP特征提取
    radius = 1  # LBP算法中范围半径的取值
    n_points = 8 * radius  # 领域像素点数

    img = local_binary_pattern(gray, n_points, radius)
    max_bins = int(img.max() + 1);

    img,_ = np.histogram(img, normed=True, bins=max_bins, range=(0, max_bins));
    return img

   
if __name__ == '__main__':
    # 将数据dump保存  下次不需要重新读取
    # train_data,test_data,image_type=load_data()
    # pickle.dump(train_data,open("train_data.pkl",'wb'))
    # pickle.dump(test_data,open("test_data.pkl",'wb'))
    # pickle.dump(image_type,open("image_type.pkl",'wb'))

    train_data=pickle.load(open("train_data.pkl",'rb'))
    test_data=pickle.load(open("test_data.pkl",'rb'))
    image_type=pickle.load(open("image_type.pkl",'rb'))

    print(train_data.keys())
    print(len(test_data))
    print(image_type)

    # cv2.namedWindow('0',cv2.WINDOW_NORMAL)

    train_feature=[]
    train_label=[]
    for class_label in train_data.keys():
        for image in train_data[class_label]:
            feature = get_LBP_feature(image)
            train_feature.append(feature)
            train_label.append(class_label)
    
    print("**************************")
    print(len(train_feature))
    print(len(train_label)) 
    train_feature=np.array(train_feature)
    train_label=np.array(train_label)
    
    print(train_feature.shape)
    print(train_label.shape)
    # 打乱顺序
    train_feature, train_label = sklearn.utils.shuffle(train_feature, train_label)
    # 划分
    x_train, x_test, y_train, y_test = train_test_split(train_feature,train_label,test_size=0.2)

    # from sklearn import tree
    # clf = tree.DecisionTreeClassifier()

    # model = OneVsOneClassifier(svm.SVC(kernel='linear',probability=True,C=1))
    # model = OneVsRestClassifier(svm.SVC(kernel='linear',probability=True))
    model = svm.SVC(kernel='linear',probability=True,gamma='auto',C=10)
    
    # model=OneVsOneClassifier(LogisticRegression(solver="liblinear",C=1)) #0.29
    # model = XGBClassifier( objective='multi：softmax')

    print("开始训练====>>>")
    model.fit(x_train,y_train)
    print("训练集准确率=====>>>>>>",model.score(x_train,y_train))
    print("验证集准确率=====>>>>>>",model.score(x_test,y_test))

    # joblib.dump(model,'hog+svmOVOgraycv.pkl')
    #重新加载model，只有保存一次后才能加载model
    # model=joblib.load('hog+svm.pkl.pkl')


    test_feature=[]
    for image in test_data:
            feature = get_LBP_feature(image)
            test_feature.append(feature)
    print(len(test_feature))
    test_feature = np.array(test_feature)


    preds = model.predict(test_feature)
    test=[]
    for i in range(preds.shape[0]):
        test.append(image_type[preds[i]])
    print(len(test))
    sample = pd.read_csv("sample_submission.csv")
    submission = pd.DataFrame({'file': sample['file'], 'species': test})
    submission.to_csv('LBP+SVM.csv', index=False)    
