from encodings import utf_8
from operator import imod
from turtle import pd
from numpy import dtype
import pandas
from sklearn.datasets import load_iris
from sklearn.decomposition import sparse_encode

# sklearn 数据集使用
def sets_demo():
    from sklearn.model_selection import train_test_split
    # 获取数据集
    iris = load_iris()

    print("鸢尾花数据集：\n",iris)
    print("查看数据集描述：\n",iris["DESCR"])
    print("查看特征值的名字：\n",iris.feature_names)
    print("查看特征值：\n",iris.data,iris.data.shape)
    
    # 数据集划分
    x_train,x_test,y_train,y_test = train_test_split(iris.data,iris.target,test_size=0.2,random_state=8)
    print(x_test,x_test.shape)
    print(y_test,y_test.shape)

# 字典特征抽取  one-hot编码
def dict_demo():
    from sklearn.feature_extraction import DictVectorizer
    data = [{"city":"北京","temperature":41},{"city":"武汉","temperature":32},{"city":"重庆","temperature":38}]

    # 1. 实例化对象
    transer = DictVectorizer() # 默认sparse = True 返回稀疏矩阵 
    # 2. 调用fit_transform()
    data_new = transer.fit_transform(data)
    print("data_new:\n",data_new)
    print("特征名字：\n",transer.get_feature_names())


# 文本特征抽取 统计样本特征词出现次数
def text_demo():
    from sklearn.feature_extraction.text import CountVectorizer
    data = ["life is short,i like python,python is easy","i want do it","i can do it","我爱中国,我爱 python"]
    # 1. 创建实例化对象
    transer = CountVectorizer()  # 参数 stop_words=["is","do"]
    # 2. 调用fit_tranform()
    data_new = transer.fit_transform(data)

    print("特征名字：\n",transer.get_feature_names())
    print("转换后数据：\n",data_new.toarray())

# 中文文本特征提取(自动分词)
def cut_word(text):
    import jieba
    split_word = " ".join(jieba.cut(text))
    return split_word
def text_chinese_demo():
    data = ["做人要活在当下因为你永远不知道明天和意外哪一个先来",
            "想要参与的朋友，可以直接写下评论，伴夏君会看到，若合适，会摘选到每天更新的句子中。",
            "坐井观天，不过一孔之见。登山望远，方知天外有天"]
    # 1. 分词处理
    data_new = []
    for i in data:
        data_new.append(i)

    from sklearn.feature_extraction.text import CountVectorizer
    # 1. 创建实例化对象
    transer = CountVectorizer()  # 参数 stop_words=["is","do"]
    # 2. 调用fit_tranform()
    data_final = transer.fit_transform(data)

    print("特征名字：\n",transer.get_feature_names())
    print("转换后数据：\n",data_final.toarray())

# 用tfidf方法进行文本特征抽取
def tifidf_demo():
    data = ["做人要活在当下因为你永远不知道明天和意外哪一个先来",
            "想要参与的朋友，可以直接写下评论，伴夏君会看到，若合适，会摘选到每天更新的句子中。",
            "坐井观天，不过一孔之见。登山望远，方知天外有天"]
    # 1. 分词处理
    data_new = []
    for i in data:
        data_new.append(i)

    from sklearn.feature_extraction.text import TfidfVectorizer
    # 1. 创建实例化对象
    transer = TfidfVectorizer()  # 参数 stop_words=["is","do"]
    # 2. 调用fit_tranform()
    data_final = transer.fit_transform(data)

    print("特征名字：\n",transer.get_feature_names())
    print("转换后数据：\n",data_final.toarray())

# 归一化
def minmax_demo():
    # 1. 获取数据
    import pandas as pd
    data = pd.read_csv("../../../datas/data.txt")
    data = data.iloc[: , :3] # 取前三列
    
    # 2. 实例化类
    from sklearn.preprocessing import MinMaxScaler
    transer = MinMaxScaler()    # 参数feature_range=[2,3] 默认0,1

    # 3。 调用fit_transform
    data_new = transer.fit_transform(data)
    print(data_new)

# 标准化
def stand_demo():
    # 1. 获取数据
    import pandas as pd
    data = pd.read_csv("../../../datas/data.txt")
    data = data.iloc[: , :3] # 取前三列
    
    # 2. 实例化类
    from sklearn.preprocessing import StandardScaler
    transer = StandardScaler()

    # 3。 调用fit_transform
    data_new = transer.fit_transform(data)
    print(data_new)

# 过滤低方差特征
def variance_demo():
    # 1. 获取数据
    import pandas as pd
    data = pd.read_csv("../../../datas/factor_returns.csv")
    data = data.iloc[:,1:-2]
    
    # 2. 实例化类
    from sklearn.feature_selection import VarianceThreshold
    transer = VarianceThreshold(threshold=5)  # threshold=5 默认为0

    # 3。 调用fit_transform
    data_new = transer.fit_transform(data)
    print(data.shape)
    print(data_new.shape)

    # 计算两个变量相关系数
    from scipy.stats import pearsonr
    r = pearsonr(data["pe_ratio"],data["pb_ratio"])
    print(r)

# 主成分分析：高维数据转为低维度数据 （损失少量信息）
def pca_demo():
    data = [[2,8,4,5],[6,3,0,8],[5,4,9,1]]
    # 1. 实例化类
    from sklearn.decomposition import PCA
    transer = PCA(n_components=0.95)  # n_components= 整数：降到的维度  浮点数：保留多少信息(0,1)
    # 2。 调用fit_transform
    data_new = transer.fit_transform(data)
    print(data_new.shape)
if __name__ == "__main__":
    # 代码1： sklearn数据集使用
    #  datasets_demo()
    # 代码2： 字典特征抽取  one-hot编码
    # dict_demo()
    # 代码3：  文本特征抽取
    # text_demo()
    # 代码4： 中文本特征提取
    # text_chinese_demo()
    # 代码5:  用tifitf进行文本特征提取
    tifidf_demo()
    # 代码6： 归一化
    # minmax_demo()
    # 代码7： 标准化
    # stand_demo()
    # 代码8： 过滤低方差特征
    # variance_demo()
    # 代码9： 主成分分析
    # pca_demo()
