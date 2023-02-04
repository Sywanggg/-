"""
 * Created with PyCharm
 * 作者: 阿光
 * 日期: 2021/12/16
 * 时间: 21:10
 * 描述:
"""
from sklearn import datasets
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import KFold, train_test_split
from sklearn.svm import SVC

# step1:读取数据
faces = datasets.fetch_olivetti_faces()

# 训练数据
x = faces.data
# 标签
y = faces.target

# 训练数据维度
print('[训练数据维度: ]', x.shape)

# step2:划分数据集
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)


# step3:计算评估指标
def calculate_metric(gt, pred):
    pred[pred > 0.5] = 1
    pred[pred < 1] = 0
    confusion = confusion_matrix(gt, pred)
    TP = confusion[1, 1]
    TN = confusion[0, 0]
    FP = confusion[0, 1]
    FN = confusion[1, 0]

    accuracy = (TP + TN) / float(TP + TN + FP + FN)
    specificity = TN / float(TN + FP)
    sensitivity = TP / float(TP + FN)

    return accuracy, specificity, sensitivity


# step4:进行交叉验证，4、5、6、7、8折
for fold in range(4, 9):
    print(f'[Fold:{fold}]')
    # 构建KFold对象用对切分数据集，为之后进行K折验证
    kf = KFold(n_splits=fold, random_state=0, shuffle=True)

    # 计算评估指标
    accuracy, specificity, sensitivity = 0.0, 0.0, 0.0
    for k, (train, test) in enumerate(kf.split(x_train, y_train)):
        x_ = x_train[train]
        y_ = y_train[train]

        x__ = x_train[test]
        y__ = y_train[test]

        model = SVC()
        model.fit(x_, y_)
        a_, b_, c_ = calculate_metric(y_, model.predict(x_))
        accuracy += a_
        specificity += b_
        sensitivity += c_

    print('准确率:', accuracy / fold)
    print('特异性:', specificity / fold)
    print('灵敏度:', sensitivity / fold)
