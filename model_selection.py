"""
 * Created with PyCharm
 * 作者: 阿光
 * 日期: 2021/12/16
 * 时间: 21:03
 * 描述:
"""

from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

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


model_names = ['LinearModel', 'DecisionTree', 'SVM', 'RandomForest', 'DNN']  # 算法名称

# step4:迭代4种算法进行验证不同模型在不同子集上的表现
for index, model in enumerate([LogisticRegression(solver='liblinear'),
                               DecisionTreeClassifier(),
                               SVC(kernel='linear', probability=True, max_iter=1000),
                               RandomForestClassifier(),
                               MLPClassifier(max_iter=1000)]):
    print(f'[{model_names[index]}]')

    model.fit(x_train, y_train)

    accuracy = model.score(x_test, y_test)

    accuracy, specificity, sensitivity = calculate_metric(y_test, model.predict(x_test))

    print('准确率:', accuracy)
    print('特异性:', specificity)
    print('灵敏度:', sensitivity)
