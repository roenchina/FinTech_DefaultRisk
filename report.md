# <center>信用违约预测</center>

|  姓名  |    学号    |
| :----: | :--------: |
| 余若涵 | 3180105412 |
| 范源颢 | 3180103574 |

格式：小四 1.25倍行距 10页以上 （意思是最后需要转成word格式？）



### 一、研究背景与意义

- [ ] TODO

估计客户的信用评级是银行风险管控非常重要的组成部分，银行借给债务人的借款如果无法按时收回，银行的资产将面临不必要的风险，同时债务人的信用也会受到影响。为此，银行需要在借款之前根据数据估计客户的还款能力，评价信用风险等级。



### 二、国内外研究现状与存在问题

- [ ] TODO



### 三、研究目标与研究内容

- [ ] TODO

在本项目中，我们的目的是：根据客户的历史表现判断客户是否有能力及时还款。



### 四、数据集分析

#### 4.1 数据集基本信息

本项目中，我们使用的是由Home Credit公司提供的贷款信息数据集。数据集包含application_train.csv和application_test.csv两个文件。训练集共提供307511条记录，每条记录包含121个特征和1个Target列；测试集包含48744条记录。

数据集中的每条记录都表示了一次贷款，Target的0/1值反映了借款人是否按时还款。121个特征中，有106个数值特征和16个标称特征，这些特征可分为以下四类(只列出部分)：

##### 1. 借款人基本信息

CODE_GENDER - 性别，

DAYS_BIRTH - 客户年龄，

CNT_CHILDREN - 子女数，

CNT_FAM_MEMBERS - 家人数，

OCCUPATION_TYPE - 职业，

……

##### 2. 借款人经济背景

FLAG_OWN_CAR - 是否拥有轿车，

FLAG_OWN_REALTY - 是否拥有不动产，

AMT_INCOME_TOTAL - 借款人总收入，

APARTMENT_AVG - 住房评分，

……

##### 3. 本次贷款相关信息

AMT_CREDIT - 信用额度，

AMT_ANNUITY - 贷款年金，

WEEKDAY_APPR_PROCESS_START - 申请贷款的星期，

HOUR_APPR_PROCESS_START - 申请贷款的小时，

……

##### 4. 其他信息

FLAG_MOBIL - 是否提供手机号码，

FLAG_CONT_MOBILE - 手机是否可打通，

REGION_RATING_CLIENT - 所在地区评分，

EXT_SOURCE_1/EXT_SOURCE_2/EXT_SOURCE_3 - 通过其他信息计算的该客户评分，

……



#### 4.2 可视化分析

以下可视化分析针对未经预处理的训练集aplication_train.csv

##### 4.2.1 Target列的分布

```python
app_train['TARGET'].value_counts()
app_train['TARGET'].astype(int).plot.hist()
```

输出：

```
0    282686
1     24825
```



<img src="./assets/target.png">

从上图可以看出，训练数据集中的违约记录数远大于未违约记录数，整个训练集是极度不平衡的，这种情况下，模型会花费更多的时间和“精力”去拟合未违约记录(target为0)，所以为了提供模型的效率，在之后我们将采取欠采样或过采样的方法来平衡数据集。



##### 4.2.2 特征与Target的相关性

###### 0) 计算特征与Target的相关性

```python
# Calculate the correlations and sort
correlations = app_train.corr()['TARGET']
correlations = correlations.drop(["TARGET"]).abs().sort_values()
```

###### 1) 10个相关性最小的特征

```python
# 10 least relevant features
correlations.head(10)
```

输出：

```
FLAG_DOCUMENT_20              0.000215
FLAG_DOCUMENT_5               0.000316
FLAG_CONT_MOBILE              0.000370
FLAG_MOBIL                    0.000534
FLAG_DOCUMENT_12              0.000756
AMT_REQ_CREDIT_BUREAU_WEEK    0.000788
AMT_REQ_CREDIT_BUREAU_HOUR    0.000930
FLAG_DOCUMENT_19              0.001358
FLAG_DOCUMENT_10              0.001414
FLAG_DOCUMENT_7               0.001520
Name: TARGET, dtype: float64
```

上面输出结果中的FLAG_DOCUMENT_X特征表示的是在贷款时是否提交某份文件，这类特征有20个，大部分相关性都很低。FLAG_MOBIL和FLAG_CONT_MOBILE表示的分别是客户是否拥有移动电话、电话是否可打通，AMT_REQ特征表示的是申请贷款前客户向信贷局进行查询的次数。

可以看出，这些特征的与Target列的相关性都在万分之一数量级，可以归为very weak级别。在之后的特征工程中我们将会试试它们是否可以衍生出足够好的新特征，或者直接删除这些特征，以减少运算量并提高预测的准确度。

###### 2) 20个相关性最大的特征

```python
# Display 20 most relevant featrues 
corrs = correlations.tail(20)
plt.figure(figsize = (10, 8))
plt.bar( x=0, bottom=corrs.index.astype(str), height=0.25, width=abs(corrs.values), orientation="horizontal")
plt.title('Feature Correlations with target')
```

<img src="./assets/corrs.png">

上图中排在前三的EXT_SOURCE特征是专业人士依据外部数据对该客户进行的评分，值域为[0, 1]，它们的相关性都超过了0.15；DAY_BIRTH表示的是客户的年龄；两个REGION_RATING特征是对客户所在地的评分，分为三档，用{1, 2, 3}表示。接下来我们将对这些特征进行可视化分析。



##### 4.2.3 客户年龄分布

```python
plt.figure(figsize = (10, 8))
# KDE plot of loans that were repaid on time
sns.kdeplot(app_train.loc[app_train['TARGET'] == 0, 'DAYS_BIRTH'] / -365, label = 'target == 0')
# KDE plot of loans which were not repaid on time
sns.kdeplot(app_train.loc[app_train['TARGET'] == 1, 'DAYS_BIRTH'] / -365, label = 'target == 1')
# Labeling of plot
plt.xlabel('Age (years)'); plt.ylabel('Density'); plt.title('Distribution of Ages');
```

<img src = "./assets/age.png">

上图中，橘线展示的是违约用户的年龄分布，蓝线为未违约用户的年龄分布。从总可以看出，年轻用户([25, 40]年龄区间内)更容易违约，而年纪较大的用户([50, 65]年轻区间内)则更可能按时付清欠款。

为了更好地分析客户年龄与违约的关系，我们以5年为一个区间宽度进行分箱处理，并用柱状图展示各个区间内的违约率。如下所示。

```python
# Age information into a separate dataframe
age_data = app_train[['TARGET', 'DAYS_BIRTH']]
age_data['YEARS_BIRTH'] = age_data['DAYS_BIRTH'] / -365
# Bin the age data
age_data['YEARS_BINNED'] = pd.cut(age_data['YEARS_BIRTH'], bins = np.linspace(20, 70, num = 11))
age_data.head(10)
# Group by the bin and calculate averages
age_groups  = age_data.groupby('YEARS_BINNED').mean()

plt.figure(figsize = (8, 8))
# Graph the age bins and the average of the target as a bar plot
plt.bar(age_groups.index.astype(str), 100 * age_groups['TARGET'])
# Plot labeling
plt.xticks(rotation = 75); plt.xlabel('Age Group (years)'); plt.ylabel('Failure to Repay (%)')
plt.title('Failure to Repay by Age Group');
```

<img src="./assets/age_group.png">

从图中可以看出，随着年龄的增高，违约率一直在下降。年纪最小的三个群体有超过10%的违约率，而年纪最大群体的违约率则低于5%. 这提示我们，年龄将会是一个非常重要的特征，在之后的特征工程中，我们也许可以基于年龄衍生出新的重要特征。



##### 4.2.4 附加特征(EXT_SOURCE)的分布

上文4.2.2的分析显示，三个EXT_SOURCE列与Target列有很大的相关性。下面探究它们的分布。

```python
plt.figure(figsize = (10, 12))

# iterate through the sources
for i, source in enumerate(['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']):
    # create a new subplot for each source
    plt.subplot(3, 1, i + 1)
    # plot repaid loans
    sns.kdeplot(app_train.loc[app_train['TARGET'] == 0, source], label = 'target == 0')
    # plot loans that were not repaid
    sns.kdeplot(app_train.loc[app_train['TARGET'] == 1, source], label = 'target == 1')
    # Label the plots
    plt.title('Distribution of %s by Target Value' % source)
    plt.xlabel('%s' % source); plt.ylabel('Density');
    
plt.tight_layout(h_pad = 2.5)
```

<img src="./assets/EXT.png">

可以看出，在以上三张图中都能很好地区分违约与未违约这两条曲线。其中EXT_SOURCE_3的效果最好，当其值较小时，有更大概率发生违约。而EXT_SOURCE_2虽然也能很好区分两条曲线，但是两曲线的整体趋势非常相似，所以也许它在模型的效果会不如另两个特征。



##### 4.2.5 地区评分(REGION_RATING)的分布

```python
for i, source in enumerate(['REGION_RATING_CLIENT', 'REGION_RATING_CLIENT_W_CITY']):
    plt.figure(figsize = (5, 3))
    tot = app_train[app_train['TARGET']!=-1].groupby(source).count()['TARGET']
    default = app_train[app_train['TARGET']==1].groupby(source).count()['TARGET']
    plt.bar(tot.index, default / tot, width=0.35, color='lightskyblue')
    
    # Label the plots
    plt.title('Distribution of %s ' % source)
    plt.xlabel('%s' % source); plt.ylabel('Default Rate');
    plt.show()
```

<img src="./assets/region1.png" style="zoom:80%;"   > <img src="./assets/region2.png" style="zoom:80%;"   >

图中柱状图表示的是违约率即违约客户占总客户的比例。两个特征都呈现出一样的趋势，即评分越高则违约率越大。可以看出这两张柱状图的趋势和数值都非常相似，所以我们推测这两个特征的相关性应该非常高，接下来进行具体分析。



##### 4.2.6 特征间的相关度

由于本数据集特征过多，不适合对所有特征进行相关度可视化，所以这一部分我们只选取了上面讨论过的六个特征进行分析。

```python
# Extract the EXT_SOURCE variables and show correlations
ext_data = app_train[['TARGET', 'EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3', 'DAYS_BIRTH', 'REGION_RATING_CLIENT', 'REGION_RATING_CLIENT_W_CITY']]
ext_data_corrs = ext_data.corr()

# Heatmap of correlations
plt.figure(figsize = (6, 6))
sns.heatmap(ext_data_corrs, cmap = plt.cm.RdYlBu_r, vmin = -0.6, annot = True, vmax = 0.6)
plt.title('Correlation Heatmap');
```

<img src="./assets/corr.png">

上面的热力图中，颜色越深表示二者之间相关度越高。可以看出两个REGION_RATING类特征有高达0.95的相关性，这两个特征都是对客户所在地区的评分，二者的差异仅在于：客户所在城市是否在考虑范围内，于是我们可以认为，它们包含的信息是基本一致的，在特征工程中我们会考虑删除这两个特征中的一个，以提高效率。此外，EXT_SOURCE_1和年龄的相关度也有0.6，这说明年龄可能是影响EXT_SOURCE_1得分的一个重要因素。



#### 4.3 数据预处理

- [ ] TODO

  编码

  缺失

  异常



### 五、研究方法与模型思路

> 这一章偏重于理论介绍

#### 5.1 特征工程

#### 5.2 模型介绍



### 六、实验与分析

> 这一章偏重于的对(五)的实现，以及对比不同实现方法的效果好坏

#### 6.1 特征工程

主要参照 [FeatureSelection](https://www.kaggle.com/willkoehrsen/introduction-to-feature-selection) 上对不同特征工程的效果测试

#### 6.2 



### 七、结论与展望

- [ ] TODO



### 参考文献
