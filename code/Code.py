#防止内存占用过大导致内核挂掉
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
import numpy as np     ##科学计算包
import pandas as pd    ##读取csv文件包
import catboost as cbt   #catboost 算法包
from lightgbm.sklearn import LGBMClassifier    #lbg 分类算法包
from xgboost import XGBClassifier     #xbg 分类算法包
from sklearn.cluster import KMeans   #kmeans  聚类算法包
import matplotlib.pyplot as plt   ##可视化包
import seaborn as sns
from sklearn.preprocessing import LabelEncoder  #标签编码包
import time   ##时间包
import warnings    ##警告包
warnings.filterwarnings('ignore')##过滤警告
##数据读取
# 加载数据
train = pd.read_csv('./data/first_round_training_data.csv')  #训练集数据
test = pd.read_csv('./data/first_round_testing_data.csv')   #测试集数据

##数据处理
#训练集数据的标签分布可视化
dit = {'Excellent':0,'Good':1,'Pass':2,'Fail':3}
# 把训练集跟测试集进行合并  这样方便数据的统一处理  有标签的为训练集 标签为空的是测试集
data = train.append(test).reset_index(drop=True)
#直接map , dataframe 类别excellent 全为 0 ，类别good 全为1 .....
train['label'] = train['Quality_label'].map(dit)
plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False
train['label'].hist()
plt.xlabel('Label')
plt.ylabel('number')
plt.title('Distribution Map of Training Set\'Label')
plt.show()
#看出来 最后一列最小，标签分布均衡
#查看数据是否用空值
print(data.isnull().sum())
#查看训练集类别数
for i in train.columns:
    print(i,len(train[i].unique()))
#测试集类别数
for i in test.columns:
    print(i,len(test[i].unique()))

##密度估计图可视化 (可以更直观的分析数据)
# 训练集 istrain== 1
# 测试集 istrain==0
train['IStrain'] = 1
test['IStrain'] = 0
# 合并数据
data = pd.concat([train, test])
# 训练集数据
train_mask = data.IStrain == 1
# 测试集数据
test_mask = data.IStrain == 0
# 对它循环进行显示
for col in test.columns:
    # 对istrain group 这两列进行排除，因为它们跟我们的可视化无关
    if col in ['IStrain', 'Group']:
        continue
        # 设置图像大小
    fig, ax = plt.subplots(1, 1, figsize=(12, 5))
    # 画出折线图
    sns.kdeplot(data.loc[train_mask, col], ax=ax, label='Train data')
    sns.kdeplot(data.loc[test_mask, col], ax=ax, label='Test data')
    # 给每一个列名为表名
    ax.set_title('name: {}'.format(col))
    plt.show()

##数据分布可视化
#对数据重置下标 生成index 列
data = data.reset_index()
#更改列名  把刚生成index列 改成 indexCol
data = data.rename(columns={'index':'indexCol'})
# 循环输出每一列的数据分布图
for col in test.columns:
    if col in ['IStrain','Group']:
        continue
    #设置图像大小（长12，宽5）
    plt.figure(figsize=(12, 5))
#     训练集为红色的部分 ，右边6000-12000 indexcol
    plt.scatter(data.loc[data.IStrain==1,'indexCol']+6000, data.loc[data.IStrain==1,col], color='g')
#     测试集为蓝色的部分 ，右边0-6000 indexcol
    plt.scatter(data.loc[data.IStrain==0,'indexCol'], data.loc[data.IStrain==0,col], color='b')
    #设置x轴的 表名
    plt.xlabel('indexCol')
    #设置表名
    plt.title(col)
    plt.show()
#通过图像观察可以发现训练集和测试集分布图中发现p5可能是类别特征（因为在p5的数值分布比较均衡，层与层之间间隔相差不大）
#因为每个特征中都出现了一定的噪声点，所以这里为了减少对于训练过程中的干扰，采用众数填充法对于每个特征类型中的噪声点进行修正
# 定义一个数据处理的函数
def Clean(train, test):
    # 标签编码
    # Excellent':,'Good':,'Pass':,'Fail':四个类别进行编码 使它变成数值型
    # 用一个字典 先封装   #用 0 1 2 3 对类别的映射
    dit = {'Excellent': 0, 'Good': 1, 'Pass': 2, 'Fail': 3}
    # 把训练集跟测试集进行合并  这样方便数据的统一处理  有标签的为训练集 标签为空的是测试集
    data = train.append(test).reset_index(drop=True)
    # 直接map , dataframe 类别excellent 全为 0 ，类别good 全为1 .....
    data['label'] = data['Quality_label'].map(dit)

    # 因为数据本身保留了很多位小数 所以进行保留7位小数 这样可以清理一些异常值
    data = np.around(data, decimals=7)

    # 数据异常值处理 ，这些数据偏离了离群点 我们对它进行异常值处理 用众数填充
    # 也可以用其它填充 ，我们这里选择用众数填充 能上分
    data['Parameter5'][data['Parameter5'] > 50] = data['Parameter5'].mode().values[0]
    data['Parameter7'][data['Parameter7'] > 5000] = data['Parameter7'].mode().values[0]
    data['Parameter8'][data['Parameter8'] > 2000] = data['Parameter8'].mode().values[0]
    # 进行特征选择 只选择有用的p5-p10列
    feature_name = ['Parameter{0}'.format(i) for i in range(5, 11)]
    # 数据处理完了 分开训练集 测试集  tr_index 训练集数据的下标
    tr_index = ~data['label'].isnull()
    X_train = data[tr_index][feature_name].reset_index(drop=True)
    # X_test 已划分好的测试集数据
    y0 = data[tr_index]['label'].reset_index(drop=True).astype(int)
    X_test = data[~tr_index][feature_name].reset_index(drop=True)

    # p5  p6 类别特征 进行训练集 测试集的统一
    for i in X_train['Parameter6'].unique():
        if i not in X_test['Parameter6'].unique():
            # 如果不统一的用众数填充
            X_train['Parameter6'][X_train['Parameter6'] == i] = X_train['Parameter6'].mode().values[0]
    # 与上面一样
    for i in X_train['Parameter5'].unique():
        if i not in X_test['Parameter5'].unique():
            X_train['Parameter5'][X_train['Parameter5'] == i] = X_train['Parameter5'].mode().values[0]

    # 添加新特征  p78
    # 我们可以对标签进行排序观察，p7 跟 p8 搭配的时候明显更好一些
    X_train['Parameter78'] = X_train['Parameter7'] + X_train['Parameter8']
    X_test['Parameter78'] = X_test['Parameter7'] + X_test['Parameter8']
    #     返回训练集，测试集 ，训练集标签，所有总数据集
    return X_train, X_test, y0, data

#Clean后的结果显示
X_train,X_test,y0,data=Clean(train,test)

#对数据重置下标 生成index 列
data = data.reset_index()
#更改列名  把刚生成index列 改成 indexCol
data = data.rename(columns={'index':'indexCol'})
# 循环输出每一列的数据分布图
for col in test.columns:
    if col in ['IStrain','Group']:
        continue
    #设置图像大小（长12，宽5）
    plt.figure(figsize=(12, 5))
#     训练集为红色的部分 ，右边6000-12000 indexcol
    plt.scatter(data.loc[data.IStrain==1,'indexCol']+6000, data.loc[data.IStrain==1,col], color='g')
#     测试集为蓝色的部分 ，右边0-6000 indexcol
    plt.scatter(data.loc[data.IStrain==0,'indexCol'], data.loc[data.IStrain==0,col], color='b')
    #设置x轴的 表名
    plt.xlabel('indexCol')
    #设置表名
    plt.title(col)
    plt.show()
X_train['IStrain'] = 1
X_test['IStrain'] = 0
#合并数据
data1 = pd.concat([X_train,X_test])
#训练集数据
train_mask = data1.IStrain==1
# 测试集数据
test_mask =  data1.IStrain==0
#对它循环进行显示
for col in X_test.columns:
    #对istrain group 这两列进行排除，因为它们跟我们的可视化无关
    if col in ['IStrain','Group']:
        continue
        #设置图像大小
    fig, ax = plt.subplots(1, 1, figsize=(12, 5))
    #画出折线图
    sns.kdeplot(data1.loc[train_mask, col], ax=ax, label='Train data')
    sns.kdeplot(data1.loc[test_mask, col], ax=ax, label='Test data')
    #给每一个列名为表名
    ax.set_title('name: {}'.format(col))
    plt.show()
#处理后的训练集中的数据更加更加贴近于测试集
#同时数据分布也较之前的有所改善
##4、生成数据
#为了能够提分，我们这里用生成数据来增加训练集的数量
#就是用一些初赛的测试集 ，因为测试集没有标签列 所以我们把预测概率高的测试集（默认预测对的）当做训练集
#因为预测两次（生成数据之后再用生成的数据再生成一遍），尽可能的多生成一些有用的数据，
#步骤：数据预处理–>特征工程–>模型调用–>有用数据选取–>在生成数据的基础上再生成一次数据
# 基于模型预测的异常值检测函数，这里我们选择的是lightgbm作为异常值检测模型
def find_outliers(model, X, y, tolerance=3):#tolerance 容差 ,这里设置的容差为3，容差就是本来是不合格的产品，预测成了优质的产品
        #本来优质的产品预测成了不合格的产品 ，这里把那些当成异常值
        #模型训练
        model.fit(X,y)
        #结果预测
        y_pred = model.predict(X)
        #求出绝对误差
        resid=abs(y-y_pred)
        #当判断模型预测出来的绝对误差大于你给的容差 ，我们就把它当做异常 获得它的下标
        outliers = resid[abs(resid)>=tolerance].index
#         返回绝对误差太大的数据下标
        return outliers

# 定义一个数据生成函数
def Generatedata(train):
    # 对数据进行处理，调用上面的clean函数
    X_train, X_test, y0, data = Clean(train, test)
    # 进行log1p 平滑数据 因为数据长尾 小的值很小 ，大的值贼大
    X_train = np.log1p(X_train)
    X_test = np.log1p(X_test)
    # 使用lgb模型查找和删除异常值
    outliers = find_outliers(LGBMClassifier(), X_train, y0)
    print(outliers)
    # 从数据中永久删除这些异常值
    X_train = X_train.drop(outliers)
    # 把y0标签数据也删除
    y0 = y0.drop(outliers)

    # 调用模型
    cbt_model = cbt.CatBoostClassifier(iterations=1450, learning_rate=0.048, verbose=1900, task_type='GPU')
    # 训练模型
    cbt_model.fit(X_train, y0)
    # 进行预测概率   predict_proba 可以对分类进行预测概率，得到的是每个类别的概率
    prediction = cbt_model.predict_proba(X_test)
    #     第三个类别下标的列表
    lis3 = []
    tr = pd.DataFrame(prediction)
    # 把预测的大概率那些数据集取出来（这里我定义概率大于85的为大概率）
    lis3.append(tr[tr[3] > 0.85].index)
    #     第二个类别下标的列表
    lis2 = []
    tr = pd.DataFrame(prediction)
    lis2.append(tr[tr[2] > 0.85].index)
    #     把test 赋给test2
    test2 = test
    #     因为第一个，第四个类别没有概率大于百分之85的 所以没有计算
    # 训练集没有group 所以删掉
    test2 = test2.drop('Group', 1)
    #     把测试集 变成训练集 给个预测的标签
    tr = test2.iloc[list(lis3[0])]
    tr['Quality_label'] = 'Fail'
    tr1 = test2.iloc[list(lis2[0])]
    tr1['Quality_label'] = 'Pass'
    #     添加到一个
    tr = tr.append(tr1)
    #     返回一个生成的训练集
    return tr
# 调用生成数据函数
tr = Generatedata(train)
# 重新导入数据
train = pd.read_csv('./data/first_round_training_data.csv')  # 训练集数据
test = pd.read_csv('./data/first_round_testing_data.csv')  # 测试集数据
# 把之前的训练集加上生成的
train = train.append(tr)
# 再进行生成一次 ，相当于迭代两次 会获得跟多的数据
tr85_85 = Generatedata(train)
# tr85_85 就是我们上面生成好的训练集 ，下边使用直接添加到训练集就好了

##特征工程与数据处理

#构造全新的训练集：训练集加上生成的训练数据
#平滑数据（这里我们用log1）
#由于p9 跟其它特征 p5 p6 p7 p8 p10 相关性很大 ，所以用预测来做有很高的准确率
#这里不用均值填充 ，因为缺失值太多 均值 众数填充价值不大，删除的话 这个是重要的特征 反而分数会更低
#加载复赛数据
train = pd.read_csv('./data/first_round_training_data.csv')  #训练集数据
train1 = pd.read_csv('./data/second_round_training_data.csv')
test = pd.read_csv('./data/second_round_testing_data.csv')
tr1=tr85_85
submit = pd.read_csv('./data/submit_example.csv')
# 把生成的数据进行去重 重点是pass 类数据 ，因为它占的比重比较多
t3=tr1[tr1['Quality_label']=='Pass']
#drop_duplicates pd 的去重函数
t3 = t3.drop_duplicates(subset=['Parameter5','Parameter6','Parameter7','Parameter8','Parameter9','Parameter10','Quality_label'], keep='first')
# 在训练集删除重复的数据
tr1=tr1.drop(t3.index,axis=0)

# 把数据添加到train
train=train.append(tr1)
train=train.append(train1)
test['Parameter1']=np.log(test['Parameter1'])
test['Parameter2']=np.log(test['Parameter2'])
test['Parameter3']=np.log(test['Parameter3'])
test['Parameter4']=np.log(test['Parameter4'])
#查看p1~4的数据分布情况
test['Parameter1'].hist()
test['Parameter2'].hist()
test['Parameter3'].hist()
test['Parameter4'].hist()

# 对数据进行数据处理 直接调用上面的clean（）函数 ，因为复赛 初赛 都一样，只换了数据
X_train,X_test,y0,data=Clean(train,test)

#因为初赛数据的p1-p5  是一些连续型数据 而且总共是6000 行数据有6000个类别所以没有使用
#复赛换了数据集 p1-p5 经过log 变换成正态分布
#使用k means 聚类获得一个p1-p5的特征，因为单独用p1-p5分数反而会下降，这也是特征选择的时候没有选择p1-p5的原因

# 聚成150个类 ，跟其它的特征差不多，给个随机种子random_state
kmeans = KMeans(n_clusters=150,random_state=456789)
X=np.array(data[['Parameter1','Parameter2','Parameter3','Parameter4']])
#用kmeans 训练
kmeans.fit(X)
# 得到聚好类的值
y_kmeans = kmeans.predict(X)
data['p1234']=pd.DataFrame(y_kmeans)

feature_name = ['Parameter{0}'.format(i) for i in range(5, 11)]
feature_name.append('p1234')
tr_index = ~data['label'].isnull()
#构建p1234特征
X_train['p1234'] = data[tr_index]['p1234']
X_test['p1234'] = data[~tr_index]['p1234']

#选择特征 这些特征都用上能够提高预测p9的准确率
X9_train=X_test[['Parameter5','Parameter6','Parameter7','Parameter8','Parameter9','Parameter10','Parameter78']][~test.isnull().T.any()]
X9_test=X_test[['Parameter5','Parameter6','Parameter7','Parameter8','Parameter10','Parameter78']][test.isnull().T.any()]

#对数据进行标签编码，因为标签是浮点型 不能做为标签
y9_train = LabelEncoder().fit_transform(X9_train['Parameter9'])
X9_train=X9_train.drop('Parameter9',1)
# 使用lgb 来预测p9 线下准确率更高
model2 = LGBMClassifier(random_state=15)
model2.fit(X9_train,y9_train)
p=model2.predict(X9_test)
#重新导入训练集数据
t=pd.read_csv('./data/second_round_training_data.csv')

#我们只用<20000的 ，其它的太大了定为异常值
t9=t[t['Parameter9']<20000]
# 我们要获得标签编码之前的数据得对它进行排序，标签编码的原理就是对数值进行排序过，然后就是它的下标+1
t9=t9.sort_values(['Parameter9'])
# zlst  p9 的所有种类数值
zlst=t9['Parameter9'].unique()

pr9=pd.DataFrame(p)
for i in range(10):
    pr9[0][pr9[0]==i]=zlst[i]
p=np.array(pr9[0])

X_test=X_test.fillna(-1)
m9=0
for i in X_test['Parameter9'].index:
    if X_test.loc[[i],['Parameter9']].values[0][0]==-1:
        X_test.loc[[i],['Parameter9']]=p[m9]
        m9+=1

#再对数据进行log1p 进行平滑
X_train=np.log1p(X_train)
X_test=np.log1p(X_test)

# 使用模型查找和删除异常值
outliers= find_outliers(LGBMClassifier(), X_train, y0)
print(outliers)
# 从数据中永久删除这些异常值
X_train=X_train.drop(outliers)
y0=y0.drop(outliers)

##模型训练
#采用catboost+XGBoost进行训练，二者分别对于训练集进行训练，最终得到的预测结果按照一定的权重比进行融合（好：差=3:1）
#使用catboost 模型
cbt_model = cbt.CatBoostClassifier(iterations=1450,learning_rate=0.048,verbose=1900,task_type='GPU')
cbt_model.fit(X_train, y0 )
# 预测测试集
prediction = cbt_model.predict_proba(X_test)
#使用xgb模型
xgbclf =XGBClassifier(num_leaves=30, reg_alpha=3, reg_lambda=10,
max_depth=28, n_estimators=200, objective='multiclass',
subsample=0.7, colsample_bytree=0.7,
random_state=15,task_type='GPU').fit(X_train, y0 )
p2 = xgbclf.predict_proba(X_test)
sub = test[['Group']]
prob_cols = [i for i in submit.columns if i not in ['Group']]
for i, f in enumerate(prob_cols):
    #对结果进行两个模型的权比融合，相对本题目比较好的给予大点的权重0.75 ，比较差的给0.25  的权重
    sub[f] = 0.25*prediction[:,i]+0.75*p2[:,i]
for i in prob_cols:
    sub[i] = sub.groupby('Group')[i].transform('mean')
sub = sub.drop_duplicates()
#对数据进行保存csv  文档
# float_format='%.3f' 保留小数后三位小数
sub.to_csv("sub.csv",index=False,float_format='%.3f')