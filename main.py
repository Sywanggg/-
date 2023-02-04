"""
 * Created with PyCharm
 * 作者: guang
 * 日期: 2021/12/15
 * 时间: 12:27
 * 描述: 使用PCA进行降维，调用SVC进行数据评估
"""

from sklearn import datasets
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

# step1:读取训练数据
faces = datasets.fetch_olivetti_faces()

# 训练数据
x = faces.data
# 标签
y = faces.target

# 训练数据维度
print('[训练数据维度: ]', x.shape)

# step2:划分数据集
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

# step3:构建支持向量机模型
clf = SVC()


# step4:训练模型
clf.fit(x_train, y_train)
print('[测试集精度: ]', accuracy_score(y_test, clf.predict(x_test)))

# step5:进行PCA降维，抽取50列特征
pca = PCA(n_components=50)

# step6:使用降维后的数据训练模型
pca.fit(x_train)

# step7:转化数据集,进行降维
x_train_pca = pca.transform(x_train)
x_test_pca = pca.transform(x_test)

# step8:构建模型
clf = SVC()
clf.fit(x_train_pca, y_train)
print('[PCA降维后精度: ]', accuracy_score(y_test, clf.predict(x_test_pca)))
