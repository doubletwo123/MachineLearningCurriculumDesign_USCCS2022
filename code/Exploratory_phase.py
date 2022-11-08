#初期阶段对于数据进行简单的数据可视化分析，之后进行特征工程以及特征提取，随后进行模型的简单选择
## 数据加载相关的包
import pandas as pd
import numpy as np

# 防止部分警告
import warnings

from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split, GridSearchCV

warnings.filterwarnings("ignore")

# 数据可视化
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# 数据的标签处理
from sklearn.preprocessing import LabelEncoder

#卡方检验
from sklearn.feature_selection import chi2
from sklearn.feature_selection import SelectKBest

#读取数据
train_data = pd.read_csv('./data/first_round_training_data.csv')

# 数据探索（数据EDA）+ 数据离散性和连续性分析
# 获取列名
col_name = train_data.drop(['Quality_label'],1).columns
Notdlts_count = []
for i in col_name:
    # 计算非重复值的个数
    Notdlts = len(train_data[i].drop_duplicates())/6000
    Notdlts_count.append(Notdlts)


plt.plot(col_name, Notdlts_count, c='r')
plt.title('非重复值的总数计算')                 # 标题
plt.xlabel('列名')                        # x轴 的轴名
plt.ylabel('非重复数据在全数据上的占比')  # y轴 的轴名
plt.xticks(rotation=45)                   # 旋转 x轴的刻度名
plt.show()

# 提取出全部的特征
unit = train_data.drop([ 'Quality_label'], 1)

# 统计及可视化数据的分布差异
for i in col_name:
    plt.hist(unit[i], bins=20)
    plt.title('%s 平均分割取值范围计数统计图'%i)
    plt.xlabel('%s范围'%i)
    plt.ylabel('值在该范围的个数')
    plt.show()
    plt.show()

# 可视化数据的离散程度--看数据的标准差
# 获取列名
col_name = unit.columns

# 计算 标准差(std)
col_std = unit.describe().T['std']

plt.plot(col_name, col_std, c='red')  # 作图
plt.title('列 - 标准差')  # 标题
plt.xlabel('列名')  # x轴 的轴名
plt.ylabel('标准差')  # y轴 的轴名
plt.xticks(rotation=90)  # 旋转 x轴的刻度名
plt.show()

#标签处理
lb = LabelEncoder()

train_data["Quality_label"] = lb.fit_transform(train_data["Quality_label"])
unit[col_name] = unit[col_name]**(1/32)
# 遍历列名
for i in col_name:
    plt.hist(unit[i], bins=20)#将数据分为20箱进行展示
    plt.title('%s 平均分割取值范围计数统计图'%i)
    plt.xlabel('%s范围'%i)
    plt.ylabel('值在该范围的个数')
    plt.show()

##去除数据的标准差
#解决办法：将标准差进行开方
# 此处选择 开4次方 (一般在2-10之间)
plt.plot(col_name, col_std**(1/4), c='g')  # 作图
plt.plot(col_name, 10*np.ones((1,20))[0], c='m', linestyle="--")
plt.title('列 - 标准差')     # 标题
plt.xlabel('列名')           # x轴 的轴名
plt.ylabel('标准差')         # y轴 的轴名
plt.xticks(rotation=90)      # 旋转 x轴的刻度名
plt.legend(['标准差','等高线：10'])
plt.show()

##因为全部的特征取值范围默认规定应该都要>0，所以可以通过log变换，同时为了不影响接近0的小数，当x->0时，将进行ln(x+1)=1的变换
# np.log() 默认底数为 e
unit[col_name] = np.log(unit[col_name] + 1)
# 计算变换后的 标准差(std)
col_log_std = unit[col_name].describe().T['std']

plt.plot(col_name, col_log_std, c='red')  # 作图
plt.title('列 - 标准差')  # 标题
plt.xlabel('列名')  # x轴 的轴名
plt.ylabel('标准差')  # y轴 的轴名
plt.xticks(rotation=90)  # 旋转 x轴的刻度名
plt.show()

##特征归一化
for i in unit.columns:
    unit[i] = (unit[i] - unit[i].min()) / (unit[i].max() - unit[i].min())
# 遍历列名
for i in col_name:
    plt.hist(unit[i], bins=20)
    plt.title('%s The statistical graph of the range of mean segmentation values after feature normalization'%i)
    plt.xlabel('%srange'%i)
    plt.ylabel('value\'number in this range')
    plt.show()

##特征选择
##使用卡方检验
# 设置卡方检验，选择k=2个最佳特征
test = SelectKBest(score_func=chi2, k=14) ##选择成最后的十四个特征
# 进行检验
fit = test.fit(unit, train_data['Quality_label'])
# 打印卡方检验值
print(fit.scores_)
train = pd.DataFrame(fit.transform(unit),columns=['V{0}'.format(i) for i in range(1, 15)])
train.head()

##算法选择
#导入数据划分函数
X_train, X_test, y_train, y_test = train_test_split(train,train_data["Quality_label"])
#集成算法中的GBoost、Lightgbm，CatBoost
#GBoost
#GridSearchCV中调教两个重要参数（实际上是要调很多个参数的。）：
# 学习率learning_rate和集合模型的个数n_estimators，这里的param_grid就是放到各种要调教的参数与备选值。
model1 = GradientBoostingClassifier() #实例化算法
#依据算法的超参，为几个重要的超参设计几个重要的点，进行网格搜索
model1 = GridSearchCV(model1,param_grid={"learning_rate":[0.1,0.01,0.001],"n_estimators":[10,100]},verbose=2)
model1.fit(X_train,y_train)
print("使用GBoost算法")
print("最好的参数",model1.best_params_)
print("模型准确率（分类准确率）",model1.score(X_test,y_test))
#Lightgbm
model2 = LGBMClassifier(verbose=2)##实例化算法
model2.fit(X_train,y_train,eval_set=[(X_test,y_test)])##算法训练（加上交叉验证集）
history_1 = model2.evals_result_ #这里把lightgbm训练过程结果保存下来
#Catboost
model3 = CatBoostClassifier()#算法实例化
model3.fit(X_train,y_train,eval_set=[(X_test,y_test)])##模型的训练
print("模型评分",model3.score(X_test,y_test))
history_2 = model3.evals_result_#这里把catboost训练过程结果保存下来

#模型2、3的训练过程比较
a = history_1['valid_0']['multi_logloss']
b = history_2['learn']["MultiClass"]
plt.plot(np.arange(len(a)), a)
plt.plot(np.arange(len(b)), b)
plt.title('lgb-cat, loss损失图像')
plt.xlabel('迭代次数')
plt.ylabel('损失值')
plt.legend(['Lightgbm','CatBoost'])
plt.show()
#从上图中可以看到Lightgbm，模型收敛速度明显快于CatBoost；且最终收敛结果后者似乎要优于前者

#模型的准确率比较
##用来保存准确率的数组变量
a = []
b = []
# 定义模型
for i in range(10):  # 循环10，重新训练模型保存每次的准确度
    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(train, train_data['Quality_label'])
    lgb_model = LGBMClassifier()
    cbt_model = CatBoostClassifier(verbose=200)
    # 训练模型
    lgb_model.fit(X_train, y_train)
    cbt_model.fit(X_train, y_train)
    a.append(lgb_model.score(X_test, y_test))
    b.append(cbt_model.score(X_test, y_test))
x = np.arange(10)
plt.plot(x, a)
plt.plot(x, b)
plt.title('模型准确率对比')
plt.xlabel('次数')
plt.ylabel('准确率')
plt.legend(['Lightgbm', 'CatBoost'])
plt.show()
#从图像的整体上来看，Lightgbm要优于CatBoost




