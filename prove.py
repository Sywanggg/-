"""
 * Created with PyCharm
 * 作者: 阿光
 * 日期: 2021/12/15
 * 时间: 14:20
 * 描述: 使用PCA进行降维,验证到底多少维最优秀
"""

import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

# step1:读取训练数据
faces = datasets.fetch_olivetti_faces('./data')

# 训练数据
x = faces.data
# 标签
y = faces.target

# step2:划分数据集
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

n_components = []  # 用于保存降维特征数
scores = []  # 用于保存不同特征维度下的得分

# step3:迭代特征数量,判断不同特征维度下的得分
for n_component in range(10, 300, 10):
    # step4:进行PCA降维，抽取50列特征
    pca = PCA(n_components=n_component, svd_solver='full')

    # step5:使用降维后的数据训练模型
    pca.fit(x_train)

    # step6:转化数据集,进行降维
    x_train_pca = pca.transform(x_train)
    x_test_pca = pca.transform(x_test)

    # step7:构建模型
    clf = SVC()
    clf.fit(x_train_pca, y_train)
    score = accuracy_score(y_test, clf.predict(x_test_pca))
    print('[PCA降维后精度: ]', score)

    # step8:将特征数、得分追加到列表用于下面画图
    n_components.append(n_component)
    scores.append(score)

# step9:绘制图像
plt.plot(n_components, scores)
plt.show()
