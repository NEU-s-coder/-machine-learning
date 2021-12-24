#coding:utf-8

'''
说明：利用python/numpy/opencv实现图像HOG特征的提取
算法思路：
算法思路:
        1)以灰度图的方式加载图片，resize到（128,64）;
        2）灰度图像gamma校正;
		3)利用一阶微分算子Sobel函数，分别计算出灰度图像X方向和Y方向上的一阶微分/梯度图像，根据得到的两幅
        梯度图像(X方向上的梯度图像和Y方向上的梯度图像)，计算出这两幅梯度图像所对应的梯度幅值图像gradient_magnitude、
        梯度方向图像gradient_angle
		4)构造(cell_x = 128/8 =16, cell_y= 64/8 =8)大小的cell图像----梯度幅值的grad_cell图像，梯度方向的ang_cell图像，
        每个cell包含有8*8 = 64个值；
		5)将每个cell根据角度值（0-180）分为9个bin，并计算每个cell中的梯度方向直方图,每个cell有9个值；
		6)每（2*2）个cell为一个block，总共15*7个block,计算每个block的梯度方向直方图，并进行归一化处理，每个block中有9*4=36个值；
		7)计算整幅图像的梯度方向直方图HOG:将计算出来的所有的Block的HOG梯度方向直方图的特征向量首尾相接组成一个维度很大的向量
        长度为：15*7*36 = 3780，
        这个特征向量就是整幅图像的梯度方向直方图特征，这个特征可用于SVM分类。
'''
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pickle
import tqdm
from main import *

#灰度图像gamma校正
# 归一化（标准化
def gamma(img):
    #不同参数下的gamma校正
    # img1 = img.copy()
    # img2 = img.copy()
    # img1 = np.power( img1 / 255.0, 0.5 )
    # img2 = np.power( img2 / 255.0, 2.2 )
    return np.power( img , 0.5 )    

#获取梯度值cell图像，梯度方向cell图像
def div( img, cell_x, cell_y, cell_w ):
    cell = np.zeros( shape = ( cell_x, cell_y, cell_w, cell_w ) )
    img_x = np.split( img, cell_x, axis = 0 )
    for i in range( cell_x ):
        img_y = np.split( img_x[i], cell_y, axis = 1 )
        for j in range( cell_y ):
            cell[i][j] = img_y [j]
    return cell

#获取梯度方向直方图图像，每个像素点有9个值
def get_bins( grad_cell, ang_cell ):
    bins = np.zeros( shape = ( grad_cell.shape[0], grad_cell.shape[1], 9 ) )
    for i in range( grad_cell.shape[0] ):
        for j in range( grad_cell.shape[1] ):
            binn = np.zeros(9)
            grad_list = np.int8( grad_cell[i,j].flatten() )#每个cell中的64个梯度值展平，并转为整数
            ang_list = ang_cell[i,j].flatten()#每个cell中的64个梯度方向展平)
            ang_list = np.int8( ang_list / 20.0 )#0-9
            ang_list[ ang_list >=9 ] = 0
            for m in range(len(ang_list)):
                binn[ang_list[m]] += int( grad_list[m] ) #不同角度对应的梯度值相加，为直方图的幅值
          #每个cell的梯度方向直方图可视化
            # N = 9
            # x = np.arange( N )
            # str1 = ( '0-20', '20-40', '40-60', '60-80', '80-100', '100-120', '120-140', '140-160', '160-180' )
            # plt.bar( x, height = binn, width = 0.8, label = 'cell histogram', tick_label = str1 )
            # for a, b in zip(x, binn):
                # plt.text( a, b+0.05, '{}'.format(b), ha = 'center', va = 'bottom', fontsize = 10 )
            # plt.show()
            bins[i][j] = binn
    return bins

#计算图像HOG特征向量，长度为 15*7*36 = 3780   
def hog( img, cell_x, cell_y, cell_w ):
    height, width = img.shape

    # 计算图像的梯度
    gradient_values_x = cv2.Sobel( img, cv2.CV_64F, 1, 0, ksize = 5 )#x方向梯度
    gradient_values_y = cv2.Sobel( img, cv2.CV_64F, 0, 1, ksize = 5 )#y方向梯度
    # 梯度幅值
    gradient_magnitude = np.sqrt( np.power( gradient_values_x, 2 ) + np.power( gradient_values_y, 2 ) )
    # 梯度方向 反正切
    gradient_angle = np.arctan2( gradient_values_x, gradient_values_y )
    # print( gradient_magnitude.shape, gradient_angle.shape )
    # plt.figure()
    # plt.subplot( 1, 2, 1 )
    # plt.imshow(gradient_angle)
    #角度转换至（0-180）
    gradient_angle[ gradient_angle > 0 ] *= 180 / 3.14
    gradient_angle[ gradient_angle < 0 ] = ( gradient_angle[ gradient_angle < 0 ] + 3.14 ) *180 / 3.14
    
    # plt.subplot( 1, 2, 2 )
    # plt.imshow( gradient_angle )
    # plt.show()

    grad_cell = div( gradient_magnitude, cell_x, cell_y, cell_w )
    ang_cell = div( gradient_angle, cell_x, cell_y, cell_w )
    bins = get_bins ( grad_cell, ang_cell )
    feature = []
    for i in range( cell_x - 1 ):
        for j in range( cell_y - 1 ):
            tmp = []
            tmp.append( bins[i,j] )
            tmp.append( bins[i+1,j] )
            tmp.append( bins[i,j+1] )
            tmp.append( bins[i+1,j+1] )
            tmp -= np.mean( tmp )
            feature.append( tmp.flatten() )
    return np.array( feature ).flatten()
                


def get_HOG_feature(img):

    # img = create_mask_for_plant(img)

    # img = segment_plant(img)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # cv2.imshow('0',img)
    # cv2.waitKey(0)
    # print( img.shape )

    resizeimg = cv2.resize( img, ( 128, 64 ), interpolation = cv2.INTER_CUBIC )
    cell_w = 8
    cell_x = int( resizeimg.shape[0] / cell_w )#cell行数
    cell_y = int( resizeimg.shape[1] / cell_w )#cell列数
    # print("每个cell的大小为{0}x{0}".format(cell_w))
    # print( 'The size of cellmap is {0}x{1} '.format( cell_x, cell_y ) )
    gammaimg = gamma( resizeimg ) 
    feature = hog( gammaimg, cell_x, cell_y, cell_w )
    # print( 'HOG_feature.size->{0}'.format(feature.shape) )
    # 大小为(3780,)
    return feature


if __name__ == '__main__':
    # 将数据dump保存  下次不需要重新读取
    train_data,test_data,image_type=load_data()
    # pickle.dump(train_data,open("train_data.pkl",'wb'))
    # pickle.dump(test_data,open("test_data.pkl",'wb'))
    # pickle.dump(image_type,open("image_type.pkl",'wb'))

    # train_data=pickle.load(open("train_data.pkl",'rb'))
    # test_data=pickle.load(open("test_data.pkl",'rb'))
    # image_type=pickle.load(open("image_type.pkl",'rb'))

    # cv2.namedWindow('0',cv2.WINDOW_NORMAL)
    


    print(train_data.keys())
    print(len(test_data))
    print(image_type)


    train_feature=[]
    train_label=[]
    for class_label in train_data.keys():
        for image in train_data[class_label]:
            feature = get_HOG_feature(image)
            train_feature.append(feature)
            train_label.append(class_label)
    print(len(train_feature))
    print(len(train_label))
    train_feature=np.array(train_feature)
    
    # train_feature =np.array(pd.DataFrame(train_feature).apply(lambda x: (x - x.mean()) / (x.std()))) #归一化
    train_label=np.array(train_label)

    # 打乱顺序
    train_feature, train_label = sklearn.utils.shuffle(train_feature, train_label)
    # 划分测试集和训练集
    x_train, x_test, y_train, y_test = train_test_split(train_feature,train_label,test_size=0.2)

    t1=pd.DataFrame(x_test)
    t2=pd.DataFrame(y_test)
    t=pd.concat([t1,t2],axis=1)
    t.to_csv("train_feature_data.csv",index=False)

    # model = OneVsOneClassifier(svm.SVC(kernel='linear',probability=True,C=0.25))
    # model = OneVsRestClassifier(svm.SVC(kernel='linear',gamma='auto',probability=True,max_iter=80000))
    # model=OneVsRestClassifier(LogisticRegression(solver="liblinear",C=10))
    # model=svm.SVC(kernel='linear',gamma='auto',probability=True,C=10)

    model=svm.SVC(kernel='linear',gamma='auto',probability=True,C=1)
    print("开始训练-->")
    model.fit(x_train,y_train)
    print("训练集准确率=====>>>>>>",model.score(x_train,y_train))
    print("验证集准确率=====>>>>>>",model.score(x_test,y_test))

    # print( len(model.estimators_) )
    # print("========",metrics.accuracy_score(train_label,model.predict(train_feature)))
    # print("========",model.score(y_train,y_test.values.flatten()))

    # joblib.dump(model,'hog+svmOVO+shuffle.pkl')
    #重新加载model，只有保存一次后才能加载model
    # model=joblib.load('hog+svm.pkl.pkl')


    test_feature=[]
    for image in test_data:
        feature = get_HOG_feature(image)
        test_feature.append(feature)
    print(len(test_feature))
    # test_feature = np.array(pd.DataFrame(test_feature).apply(lambda x: (x - x.mean()) / (x.std()))) #归一化



    preds = model.predict(test_feature)
    test=[]
    for i in range(preds.shape[0]):
        test.append(image_type[preds[i]])
    print(len(test))
    sample = pd.read_csv("sample_submission.csv")
    submission = pd.DataFrame({'file': sample['file'], 'species': test})
    submission.to_csv('HOG+SVM.csv', index=False)    


