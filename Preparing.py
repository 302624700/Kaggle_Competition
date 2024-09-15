import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


train = pd.read_csv('data\\train.csv', index_col='id')  # 共188533辆二手车

train_y = train.price  # 预测变量
train.drop(['price'], axis=1, inplace=True)

train.info()  # 查看各变量类型
train.isnull().sum()  # 查看缺失值数量
train.describe()  # 描述统计数值型变量


# brand
brand = train.brand  # 57个品牌
bb = brand.value_counts()
sum(bb.values[:30]) / sum(bb.values)  # 前30个品牌占比95.73%

plt.pie(bb[:30], labels=bb.index[:30], autopct='%0.1f%%')  # 数量前30二手车品牌占比
plt.axis('square')
plt.show()

del(bb, brand)


# model_year
model_year = train.model_year  # 34个年份
mb = model_year.value_counts()

plt.pie(mb, labels=mb.index, autopct='%0.1f%%', pctdistance=0.8)
plt.axis('square')
plt.show()

del(mb, model_year)


# milage
milage = train.milage  # 里程数
sns.distplot(milage, bins=20, kde=False)
plt.show()

del milage


# fuel_type
fuel_type = train.fuel_type  # 7种燃料类型
fb = fuel_type.value_counts()

plt.pie(fb, autopct='%0.1f%%', pctdistance=0.8)
plt.legend(fb.index,loc=3)
plt.axis('square')
plt.show()

del(fb, fuel_type)


# engine
engine = train.engine  # 共1117种发动机
eb = engine.value_counts()

del (eb, engine)


# transmission
transmission = train.transmission  # 共52种变速
tb = transmission.value_counts()
transmission_rate = tb.values / tb.values.sum()  # 将不同变速计数转化为百分比形式

plt.bar(x=tb.index[:20], height=transmission_rate[:20], color='gray')  # 显示占比前20的变速箱条形图
plt.xticks(rotation=90)
plt.show()

del(tb, transmission, transmission_rate)


# ext_col
ext_col = train.ext_col
eb = ext_col.value_counts()

del(eb, ext_col)


# int_col
int_col = train.int_col
ib = int_col.value_counts()

del(ib, int_col)


# accident
accident = train.accident  # 仅有两个取值,有无已知事故
ab = accident.value_counts()  # 144514无,41567有

del(ab, accident)


# clean_title
clean_title = train.clean_title  # 仅有一个取值, Yes
cb = clean_title.value_counts()  # 167114 Yes, 21419 缺失值
del(cb, clean_title)
