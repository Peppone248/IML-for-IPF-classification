import pandas
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from useful_methods import features_encoding

pandas.set_option('display.max_columns', None)
data = pandas.read_csv('new Data Set Fibrosi.csv')
df = pandas.DataFrame(data)

stats = df.describe()
print(stats)

total_columns = df.columns
num_col = df._get_numeric_data().columns
cat_col = list(set(total_columns) - set(num_col))

null_df = df.apply(lambda x: sum(x.isnull())).to_frame(name='count')
print(null_df)
plt.plot(null_df.index, null_df['count'])
plt.xticks(null_df.index, null_df.index, rotation=60, horizontalalignment='right')
plt.xlabel('column names')
plt.title('Distribuzione valori mancanti')
plt.margins(0.1)
plt.show()

df = df.drop('ID Lab', axis=1)
gender = df['Genere'].values

features_encoding(df)

unique, counts = np.unique(gender, return_counts=True)
plt.pie(counts, labels=['M','F'], autopct=lambda p: '{:.2f}%  ({:,.0f})'.format(p, p * sum(counts) / 100))
plt.show()

"""
skew = {}
kurt = {}
for i in num_col:
    # to skip columns for plotting
    if i in ["num_orders"]:
        continue
    skew[i] = df[i].skew()
    kurt[i] = df[i].kurt()
print(skew)


plt.plot(list(skew.keys()), list(skew.values()))
plt.xticks(rotation=45, horizontalalignment='right')
plt.show()

print(kurt)
plt.plot(list(kurt.keys()), list(kurt.values()))
plt.xticks(rotation=45, horizontalalignment='right')
plt.show()
"""

"""for i in cat_col:
    if i in ['source']:
        continue
    plt.figure(figsize=(10, 5))
    chart = sns.countplot(data=df, x=i, palette='Set1')
    chart.set_xticklabels(chart.get_xticklabels(), rotation=45)
    plt.show()"""

""" 
The skew() function used to calculate skewness in data. 
It represents the shape of the distribution. 
Skewness can be quantified to define the extent to which a distribution differs from a normal distribution.
    
The kurt() function used to calculate kurtosis in data. 
Kurtosis is the measure of thickness or heaviness of the distribution. 
It represents the height of the distribution.

"""

"""df = df.iloc[:, 6:16]
df.boxplot(figsize=(20,15))
plt.semilogy(labels=df.values)
plt.show()"""

"""sns.boxplot(data=df.iloc[:, 6:16], palette='Set2')
plt.semilogy()
plt.show()"""

mat_correlation = df.corr()
plt.figure(figsize=(13, 6))
sns.heatmap(mat_correlation, vmax=1, annot=True, linewidths=.5)
plt.xticks(rotation=30, horizontalalignment='right')
plt.show()

mean_lin = df['Neu%'].mean()
print('Lin% features mean: ', mean_lin)


"""select_hp = df.loc[df['Patologia'] == 'HP']
median_hp_dlco = select_hp['DLCO'].median()
median_hp_neu = select_hp['Neu%'].median()
median_hp_kl6 = select_hp['2-DDCT KL-6'].median()
median_hp_dlco_va = select_hp['DLCO/VA'].median()
print('hp: ', median_hp_dlco)
print('hp: ', median_hp_kl6)
print('dlco_va', median_hp_dlco_va)

select_nsip = df.loc[df['Patologia'] == 'NSIP']
median_nsip_dlco = select_nsip['DLCO'].median()
median_nsip_fev1 = select_nsip['FEV1%'].median()
median_nsip_kl6 = round(select_nsip['2-DDCT KL-6'].mean(), 2)
print('nsip:', median_nsip_dlco)
print('nsip:', median_nsip_kl6)

select_ipf = df.loc[df['Patologia'] == 'IPF']
median_ipf_fev1 = select_ipf['FEV1%'].median()
median_ipf_dlco = select_ipf['DLCO'].median()
median_ipf_kl6 = select_ipf['2-DDCT KL-6'].median()
print('ipf:', median_ipf_dlco)
print('ipf:', median_ipf_fev1)

df.loc[11, 'DLCO'] = median_hp_dlco
df.loc[39, 'DLCO'] = median_nsip_dlco
df.loc[39, 'FEV1%'] = median_nsip_fev1
df.loc[9, 'Neu%'] = median_hp_neu
df.loc[25, 'FEV1%'] = median_ipf_fev1
df.loc[23, '2-DDCT KL-6'] = median_ipf_kl6
df.loc[37, '2-DDCT KL-6'] = median_nsip_kl6

print(df)

new_dataset = df.to_csv('new Data Set Fibrosi.csv', index=False)"""

