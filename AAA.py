from scipy.sparse.construct import rand
from sift_SGB import *
from color import *
from HOG_cv import *


data_path = "."
categories = ['Black-grass', 'Charlock', 'Cleavers', 'Common Chickweed', 'Common wheat', 'Fat Hen',
                'Loose Silky-bent',
                'Maize', 'Scentless Mayweed', 'Shepherds Purse', 'Small-flowered Cranesbill', 'Sugar beet']
CATE2ID = {v: k for k, v in enumerate(categories)}
train_image_paths, test_image_paths, train_labels = get_image_paths(data_path, categories)
trian_image_labels_id = [CATE2ID[x] for x in train_labels]
print("开始建立词汇表===========================>")
if not os.path.exists('vocab.pkl'):
    vacob = build_vocabulary(train_image_paths, 50, 100)
    pickle.dump(vacob, open('vocab.pkl', 'wb'))
else:
    vacob = pickle.load(open('vocab.pkl', 'rb'))
print(vacob.shape)
print("词汇表建立完成===========================>")
print("开始提取训练集BOW描述===========================>")
if not os.path.exists('train_X.pkl'):
    train_X = get_train_feat(train_image_paths, vacob, 50)
    pickle.dump(train_X, open('train_X.pkl', 'wb'))
else:
    train_X = pickle.load(open('train_X.pkl', 'rb'))
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
    test_X = get_train_feat(test_image_paths, vacob, 50)
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
model = svm.SVC(kernel='linear', probability=True, gamma='auto', C=72).fit(train, train_y)
# 0.6402 model = KNeighborsClassifier(n_neighbors=5).fit(train,train_y)
# model = make_pipeline(StandardScaler(),
#          SGDClassifier(max_iter=1000, tol=1e-3)).fit(train,train_y)
print("训练结束===========================>")
print("训练集准确率=====>>>>>>", model.score(train, train_y))
print("验证集准确率=====>>>>>>", model.score(x_test, y_test))
print("开始预测===========================>")
preds1 = model.predict(test)
print("预测结束===========================>")









# color
# *******************************
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
preds2 = model.predict(test_X)
print("预测结束===========================>")










# hog
# **********************************************************
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

#定义对象hog，同时输入定义的参数，剩下的默认即可
winSize = (128,128)
blockSize = (64,64)
blockStride = (32,32)
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
# pca = PCA(n_components=500)  # 自动选择特征个数  'mle'
# print("**************************")
# pca.fit(train_feature)
# print("**************************")
# print("降维前train.shape:{0}".format(train_feature.shape))
# train_feature = pca.transform(train_feature)
# print("降维后train.shape:{0}".format(train_feature.shape))


# 打乱顺序
train_feature, train_label = sklearn.utils.shuffle(train_feature, train_label)
# 划分
x_train, x_test, y_train, y_test = train_test_split(train_feature,train_label,test_size=0.2)

# model = OneVsOneClassifier(svm.SVC(kernel='linear',probability=True,C=1))
# model = OneVsRestClassifier(svm.SVC(kernel='linear',probability=True))
model = svm.SVC(kernel='linear',probability=True,gamma='auto',C=0.5)
# model=  SGDClassifier(tol=1e-3)
# model=OneVsOneClassifier(LogisticRegression(solver="liblinear",C=1)) #0.29
# model = XGBClassifier( objective='multi：softmax')

print("开始训练====>>>")
if not os.path.exists('HOG+SVM.pkl'):
    model.fit(x_train,y_train)
    joblib.dump(model,'HOG+SVM.pkl')
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

# print("降维前train.shape:{0}".format(test_feature.shape))
# test_feature = pca.transform(test_feature)
# print("降维后train.shape:{0}".format(test_feature.shape))

preds3 = model.predict(test_feature)
test1=[]
for i in range(preds3.shape[0]):
    test1.append(image_type[preds3[i]])
print(len(test1))
sample = pd.read_csv("sample_submission.csv")
submission = pd.DataFrame({'file': sample['file'], 'species': test1})
submission.to_csv('HOG+SVM.csv', index=False)    






test=[]
for i in range(preds1.shape[0]):
    temp = 0
    if preds1[i]==preds2[i]:
        temp = preds2[i]
    elif preds3[i] ==preds1[i] or preds3[i] ==preds2[i]:
        temp = preds3[i]
    else:
        temp = random.choice([ preds1[i],preds2[i] ])
    test.append(image_type[temp])
print(len(test))
sample = pd.read_csv("sample_submission.csv")
submission = pd.DataFrame({'file': sample['file'], 'species': test})
submission.to_csv('AAA.csv', index=False)   