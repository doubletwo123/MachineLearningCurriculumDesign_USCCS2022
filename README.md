## 题目：离散制造过程中典型工件的质量符合率预测
- 背景：在高端制造领域，随着数字化转型的深入推进，越来越多的数据可以被用来分析和学习，进而实现制造过程中重要决策和控制环节的智能化，例如生产质量管理。从数据驱动的方法来看，生产质量管理通常需要完成质量影响因素挖掘及质量预测、质量控制优化等环节，本赛题将关注于第一个环节，基于对潜在的相关参数及历史生产数据的分析，完成质量相关因素的确认和最终质量符合率的预测。在实际生产中，该环节的结果将是后续控制优化的重要依据。
- 任务
由于在实际生产中，同一组工艺参数设定下生产的工件会出现多种质检结果，所以我们针对各组工艺参数定义其质检标准符合率，即为该组工艺参数生产的工件的质检结果分别符合优、良、合格与不合格四类指标的比率。相比预测各个工件的质检结果，预测该质检标准符合率会更具有实际意义。 本赛题要求参赛者对给定的工艺参数组合所生产工件的质检标准符合率进行预测。
- 声明：本赛题要求参赛者对给定的工艺参数组合所生产工件的质检标准符合率进行预测。
- 实验提纲：
  + 初期进行算法模型的对比选择：
  + 采用GBDT[1]（Gradient Boosting Decision Tree）作为参照，之后又选择了两种不同的框架，对于结果进行了对比。
  + Lightgbm[2] （Light Gradient Boosting Machine）
  + CatBoost[3][4] （Categorical boosting）
  + 基本步骤：数据预处理、特征分析与选择、算法选择（针对提取到的特征值，选择合适的算法模型）、编写完整代码并求得对应的输出结果。
- References；
   >  [1] Tianqi Chen,Carlos Guestrin.XGBoost: A Scalable Tree Boosting System[（论文详解）](https://blog.csdn.net/meiyh3/article/details/127156523?ops_request_misc=%257B%2522request%255Fid%2522%253A%2522166782863016782425112494%2522%252C%2522scm%2522%253A%252220140713.130102334..%2522%257D&request_id=166782863016782425112494&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2~all~sobaiduend~default-1-127156523-null-null.142%5ev63%5econtrol,201%5ev3%5eadd_ask,213%5ev1%5econtrol&utm_term=XGBoost%E8%AE%BA%E6%96%87&spm=1018.2226.3001.4187)</br>
   >  [2] [LightGBM 中文文档](https://lightgbm.apachecn.org/#/?id=lightgbm-%e4%b8%ad%e6%96%87%e6%96%87%e6%a1%a3)</br>
   >  [3] Anna Veronika Dorogush, Vasily Ershov, Andrey Gulin. CatBoost: gradient boosting with categorical features support, Workshop on ML Systems at NIPS 2017[(论文地址)](http://learningsys.org/nips17/assets/papers/paper_11.pdf)</br>
   >  [4] Liudmila Prokhorenkova, Gleb Gusev, Aleksandr Vorobev, Anna Veronika Dorogush, Andrey Gulin. CatBoost: unbiased boosting with categorical features, NeurIPS, 2018[(论文地址)](https://arxiv.org/pdf/1706.09516.pdf)<br>



