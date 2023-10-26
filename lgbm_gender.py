import pandas
import dice_ml
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import shap
import numpy as np
import scikitplot as skplot
import lightgbm
from lightgbm.sklearn import LGBMClassifier
from sklearn.model_selection import GridSearchCV, LeaveOneOut, cross_val_score
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import MinMaxScaler
from useful_methods import plot_conf_matrix, features_encoding, GSCV_tuning_model, shap_global_charts


pandas.set_option('display.max_columns', None)
data = pandas.read_csv('new Data Set Fibrosi.csv')
df = pandas.DataFrame(data)

df_men = df[df['Genere'] == 'M']
df_women = df[df['Genere'] == 'F']

features_encoding(df_women)
features_encoding(df_men)

df_women = df_women.drop('Patologia', axis=1)
df_men = df_men.drop('Patologia', axis=1)
# df_men = df_men.dropna()

feature_cols_gender = ['2-DDCT KL-6', '2-DDCT MIR21', '2-DDCT MIR92A', 'FVC%', 'FEV1%']

classes = ['ALTRO', 'IPF']

X_not_converted = df[feature_cols_gender]
X = df_men[feature_cols_gender].values
X_w = df_women[feature_cols_gender].values
y = df_men['IPFVSALTRO'].values
y_w = df_women['IPFVSALTRO'].values
X_id = df_men['ID Lab'].values

cv = LeaveOneOut()
cv.get_n_splits(X)
y_true, y_pred = list(), list()
X_train, X_test = [], []
y_train, y_test = [], []
