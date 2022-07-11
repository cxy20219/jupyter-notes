"""KNN算法
    KNN 特点： 
       1. 在训练前，应该进行无量纲化处理(否则容易受到较大值的影响)
       1. k取得过小，容易受到异常点影响
       2. k取得过大，容易受到样本不均衡影响
    优点：简单易实现，无需训练
    缺点：
       1. k值需要合适
       2. 对测试样本分类时计算量大，内存开销大
    
    适用场景:少量数据

"""
# KNN 算法对鸢尾花分类
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

    # 用训练集的标准来标准化测试集
    x_test = transer.transform(x_test)

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


"""朴素贝叶斯算法
    朴素？ 特征与特征之间是相互独立的
    贝叶斯？贝叶斯公式

    防止计算出来概率为零，引入拉普拉斯平滑系数

    优点：
        1. 有稳定的分类效率
        2. 对缺失数据不太敏感，常用于文本分类
        3. 分类准确度高，速度快
    缺点：
        采用样本属性独立假设，如果特征属性有关联时，效果不好
    
    应用场景：文本分类

"""
# 用朴素贝叶斯算法对新闻进行分类
def nb_news():
    # 1. 获取数据集
    from  sklearn.datasets import fetch_20newsgroups
    news = fetch_20newsgroups(subset="all") #subset 默认是train 只调用训练集

    # 2. 划分数据集
    from sklearn.model_selection import train_test_split
    x_train,x_test,y_train,y_test = train_test_split(news.data,news.target,random_state=8)

    # 3. 特征工程:文本特征抽取
    from sklearn.feature_extraction.text import TfidfVectorizer
    transfer = TfidfVectorizer()
    x_train = transfer.fit_transform(x_train)
    x_test = transfer.transform(x_test)

    # 4. 朴素贝叶斯算法预估流程 
    from sklearn.naive_bayes import MultinomialNB
    estimator = MultinomialNB(alpha=1.0) # alpha默认1.0
    estimator.fit(x_train,y_train)
    # 5. 模型评估 
    # 法一 直接比对真实值和预测值
    y_predict =  estimator.predict(x_test)
    print("y_predict:\n",y_predict)
    print("直接比对真实值和预测值：\n",y_predict == y_test)
    
    # 法二 计算准确率
    score = estimator.score(x_test,y_test)
    print("准确率：\n",score)

"""决策树算法
    决策树划分依据之一：信息增益

    优点：
        可视化
    缺点：
        可以创建复杂的树，但是不能很好地推广(容易过拟合)
    
    改进：
        减枝(API已实现)
        随机深林

"""
# 用决策树树对鸢尾花进行分类
def decision_iris():
    # 1. 获取数据
    from sklearn.datasets import load_iris
    iris = load_iris()

    # 2. 划分数据集
    from sklearn.model_selection import train_test_split
    x_train,x_test,y_train,y_test = train_test_split(iris.data,iris.target,random_state= 8)

    # 3. 使用决策树预估器
    from sklearn.tree import DecisionTreeClassifier
    # entropy 以信息增益为依划分据 默认为gini
    # max_depth 树深度设置，使不会过拟合
    estimator = DecisionTreeClassifier(criterion="entropy",max_depth=4) 
    estimator.fit(x_train,y_train)
    # 4. 模型评估
    # 法一 直接比对真实值和预测值
    y_predict =  estimator.predict(x_test)
    print("y_predict:\n",y_predict)
    print("直接比对真实值和预测值：\n",y_predict == y_test)
    
    # 法二 计算准确率
    score = estimator.score(x_test,y_test)
    print("准确率：\n",score)

    # 对决策树可视化 导出为dot格式
    from sklearn.tree import export_graphviz
    export_graphviz(estimator,out_file="iris_tree.dot",feature_names=iris.feature_names)


# 随机森林
"""
    集成学习的一种方法  
    集成学习：训练多个分类器，由多个分类器的众数决定  
    随机：  
    1. 训练集随机
        * bootstrap:随机有放回抽样
    2. 特征随机
        * 从大M个特征中随机抽取m个特征  (M>>m)
        * 能实现降维
        * 能使好的结果脱颖而出

    适用场景：
    1. 在当前所有算法中具有极好的准确率
    2. 能够有效运行在大数据集上，处理高维特征输入样本，而且不需要降维
    3. 能评估各个特征在分类问题上的重要性
"""
def RandomForest():

    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import GridSearchCV    
    estimator2 = RandomForestClassifier()
    # 1. 获取数据
    from sklearn.datasets import load_iris
    iris = load_iris()

    # 2. 划分数据集
    from sklearn.model_selection import train_test_split
    x_train,x_test,y_train,y_test = train_test_split(iris.data,iris.target,random_state= 8)
    # 3.交叉验证与网格搜索
    param_dict = {
        "n_estimators" : [120,200,300,500,800,1200],
        "max_depth" : [4,5,8,15,25,30]
    }
    estimator2 = GridSearchCV(estimator2,param_grid=param_dict,cv=10)  # cv: 交叉验证的轮数
    estimator2.fit(x_train,y_train)

    # 4. 模型评估
    # 法一 直接比对真实值和预测值
    y_predict =  estimator2.predict(x_test)
    print("y_predict:\n",y_predict)
    print("直接比对真实值和预测值：\n",y_predict == y_test)

    # 法二 计算准确率
    score = estimator2.score(x_test,y_test)
    print("准确率：\n",score)
if __name__ == "__main__":
    # 代码一：  KNN 算法对鸢尾花分类
    # Knn_iris()
    # 代码二： 模型调优：添加网格搜索和交叉验证
    # Knn_iris_gscv()
    # 代码三： 朴素贝叶斯算法对新闻分类
    # nb_news()
    # 代码四： 用决策树对鸢尾花分类
    decision_iris()