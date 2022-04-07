# KNN 算法对鸢尾花分类
from matplotlib.pyplot import step


def Knn_iris(): 
    # 1. 获取数据
    from sklearn.datasets import load_iris
    iris = load_iris()

    # 2. 划分数据集
    from sklearn.model_selection import train_test_split
    x_train,x_test,y_train,y_test = train_test_split(iris.data,iris.target,random_state= 8)
    # 3. 特征工程 ： 标准化
    from sklearn.preprocessing import StandardScaler
    transer = StandardScaler()
    x_train = transer.fit_transform(x_train)
    x_test = transer.fit_transform(x_test)

    # 4. KNN算法预估器
    from sklearn.neighbors import KNeighborsClassifier
    estimator = KNeighborsClassifier(n_neighbors=9)
    estimator.fit(x_train,y_train)

    # 5. 模型评估
    # 法一 直接比对真实值和预测值
    y_predict =  estimator.predict(x_test)
    print("y_predict:\n",y_predict)
    print("直接比对真实值和预测值：\n",y_predict == y_test)
    
    # 法二 计算准确率
    score = estimator.score(x_test,y_test)
    print("准确率：\n",score)

# 模型调优：添加网格搜索和交叉验证
def Knn_iris_gscv():
    
    # 1. 获取数据
    from sklearn.datasets import load_iris
    iris = load_iris()

    # 2. 划分数据集
    from sklearn.model_selection import train_test_split
    x_train,x_test,y_train,y_test = train_test_split(iris.data,iris.target,random_state= 8)
    # 3. 特征工程 ： 标准化
    from sklearn.preprocessing import StandardScaler
    transer = StandardScaler()
    x_train = transer.fit_transform(x_train)
    x_test = transer.fit_transform(x_test)

    # 4. KNN算法预估器
    from sklearn.neighbors import KNeighborsClassifier
    estimator = KNeighborsClassifier(n_neighbors=9)

    # 5. 加入网格搜索和交叉验证
    from sklearn.model_selection import GridSearchCV
    param_dict = {"n_neighbors" : [i for i in range(1,20,2)]}
    estimator = GridSearchCV(estimator,param_grid=param_dict,cv=10)  # cv: 交叉验证的轮数
    estimator.fit(x_train,y_train)

    # 5. 模型评估
    # 法一 直接比对真实值和预测值
    y_predict =  estimator.predict(x_test)
    print("y_predict:\n",y_predict)
    print("直接比对真实值和预测值：\n",y_predict == y_test)
    
    # 法二 计算准确率
    score = estimator.score(x_test,y_test)
    print("准确率：\n",score)

    # 最佳参数： estimator.best_params_
    print("最佳参数：\n",estimator.best_params_)

    # 最佳结果：estimator.best_score_
    print("最佳结果：\n",estimator.best_score_)

    # 最佳估计器：estimator.best_estimator_
    print("最佳估计器：\n",estimator.best_estimator_)

    # 交叉验证结果：estimator.cv_results_
    print("交叉验证结果：\n",estimator.cv_results_)
if __name__ == "__main__":
    # 代码一：  KNN 算法对鸢尾花分类
    # Knn_iris()
    # 代码二： 模型调优：添加网格搜索和交叉验证
    Knn_iris_gscv()