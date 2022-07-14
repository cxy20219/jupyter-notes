"""
    正规方程
    适用场景：特征较少的场景
    优点：能直接计算出一个较好的结果
    缺点：时间复杂度大，特征较多较复杂时求解速度很慢
"""
from sklearn.linear_model import LogisticRegression


def Linear1():
    # 1. 获取数据
    from sklearn.datasets import load_boston
    boston = load_boston()

    # 2. 划分数据集
    from sklearn.model_selection import train_test_split
    x_train,x_test,y_train,y_test = train_test_split(boston.data,boston.target,random_state=8)

    # 3. 标准化
    from sklearn.preprocessing import StandardScaler
    transfer  = StandardScaler()
    x_train = transfer.fit_transform(x_train)
    x_test = transfer.transform(x_test)

    # 4. 预估器
    from sklearn.linear_model import LinearRegression
    estimator = LinearRegression()
    estimator.fit(x_train,y_train)

    # 5. 得出模型
    print("权重系数为:",estimator.coef_)
    print("偏置为:",estimator.intercept_)

    # 6. 模型评估
    from sklearn.metrics import mean_squared_error
    y_predict = estimator.predict(x_test)
    error = mean_squared_error(y_test,y_predict)

    print("均方误差为:",error)


"""
    梯度下降
    适用场景:普遍适用
"""
def Linear2():
    # 1. 获取数据
    from sklearn.datasets import load_boston
    boston = load_boston()

    # 2. 划分数据集
    from sklearn.model_selection import train_test_split
    x_train,x_test,y_train,y_test = train_test_split(boston.data,boston.target,random_state=8)

    # 3. 标准化
    from sklearn.preprocessing import StandardScaler
    transfer  = StandardScaler()
    x_train = transfer.fit_transform(x_train)
    x_test = transfer.transform(x_test)

    # 4. 预估器
    from sklearn.linear_model import SGDRegressor
    estimator = SGDRegressor(eta0=0.01,max_iter=10000) # 学习率 迭代次数
    estimator.fit(x_train,y_train)

    # 5. 得出模型
    print("权重系数为:",estimator.coef_)
    print("偏置为:",estimator.intercept_)

    # 6. 模型评估
    from sklearn.metrics import mean_squared_error
    y_predict = estimator.predict(x_test)
    error = mean_squared_error(y_test,y_predict)

    print("均方误差为:",error)


"""
    解决过拟合与欠拟合
    正则化方法：
        1. L1正则化(LASS0)
            可以使其中一些权重w为0，削弱某个特征的影响

        2. L2正则化(Ridge)岭回归
            可以使其中一些权重w接近于0，削弱某个特征的影响
            正则化力度越小，权重系数越大
            正则化力度越大，权重系数越小
"""
def Linear3():
    # 1. 获取数据
    from sklearn.datasets import load_boston
    boston = load_boston()

    # 2. 划分数据集
    from sklearn.model_selection import train_test_split
    x_train,x_test,y_train,y_test = train_test_split(boston.data,boston.target,random_state=8)

    # 3. 标准化
    from sklearn.preprocessing import StandardScaler
    transfer  = StandardScaler()
    x_train = transfer.fit_transform(x_train)
    x_test = transfer.transform(x_test)

    # 4. 预估器
    from sklearn.linear_model import Ridge
    estimator = Ridge(alpha=0.1,max_iter=10000) # 正则化力度 迭代次数
    estimator.fit(x_train,y_train)

    # 5. 得出模型
    print("权重系数为:",estimator.coef_)
    print("偏置为:",estimator.intercept_)

    # 6. 模型评估
    from sklearn.metrics import mean_squared_error
    y_predict = estimator.predict(x_test)
    error = mean_squared_error(y_test,y_predict)

    print("均方误差为:",error)



"""
    逻辑回归与二分类
    将线性回归输出带入到激活函数,输出[0,1]区间的一个概率值
"""
def LogicalClass():
    import pandas as pd
    import numpy as np
    column_name =[
    "Sample code number","Clump Thickness","Uniformity of Cell Size",
    "Uniformity of Cell Shape","Marginal Adhesion","Single Epithelial Cell Size",
    "Bare Nuclei","Bland Chromatin","Normal Nucleoli","Mitoses","Class"
    ]
    data = pd.read_csv("../../../datas/breast_cancer/breast-cancer-wisconsin.data",names=column_name)
    # 替换缺失值
    data = data.replace(to_replace='?',value=np.nan)
    # 删除缺失样本
    data.dropna(inplace=True)
    data.isnull().any()
    x = data.iloc[:,1:-1]
    y = data["Class"]

    # 划分数据集
    from sklearn.model_selection import train_test_split
    x_train,x_test,y_train,y_test = train_test_split(x,y,random_state=8)

    # 标准化
    from sklearn.preprocessing import StandardScaler
    transfer = StandardScaler()
    x_train = transfer.fit_transform(x_train)
    x_test = transfer.transform(x_test)

    # 预估器
    from sklearn.linear_model import LogisticRegression
    estimator = LogisticRegression()
    estimator.fit(x_train,y_train)

    # 模型评估
    # 法一 直接比对真实值和预测值
    y_predict =  estimator.predict(x_test)
    print("y_predict:\n",y_predict)
    print("直接比对真实值和预测值：\n",y_predict == y_test)

    # 法二 计算准确率
    score = estimator.score(x_test,y_test)
    print("准确率：\n",score)

    # 精确率与召回率
    from sklearn.metrics import classification_report
    report = classification_report(y_test,y_predict,labels=[2,4],target_names = ["良性","恶性"])
    print(report)

    # AUC指标
    # y_true必须 为0(反例),1(正例)
    y_true = np.where(y_test>3,1,0)
    from sklearn.metrics import roc_auc_score
    Auc = roc_auc_score(y_true,y_predict)
    print("AUC指标:",Auc)

if __name__ == "__main__":
    # 正规方程
    # Linear1()
    # 梯度下降
    # Linear2()
    # 岭回归
    # Linear3()
    # 逻辑回归实现分类
    LogicalClass()