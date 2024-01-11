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
from functools import reduce


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

for train_ix, test_ix in cv.split(X):
    # split data
    X_train, X_test = X[train_ix, :], X[test_ix, :]
    y_train, y_test = y[train_ix], y[test_ix]
    model = LGBMClassifier(random_state=42, verbose=-1, min_data_leaf=2)
    model.fit(X_train, y_train)
    # evaluate model
    ytemp = model.predict(X_test)
    # store
    y_true.append(y_test[0])
    y_pred.append(ytemp[0])
    print(classification_report(y_true, y_pred, zero_division=1))
    scores = cross_val_score(model, X, y, scoring='accuracy', cv=cv, n_jobs=-1)
    accuracy = accuracy_score(y_true, y_pred)
    print("Accuracy: %.2f%%" % (accuracy * 100.0))
    print('Best score: {}'.format(scores))
    print("real value ", y_true)
    print("predicted ", y_pred)

_true, y_pred = list(), list()
X_train, X_test = [], []
y_train, y_test = [], []

GSCV_tuning_model(X, y, cv, parameters={
    'boosting_type': ['goss', 'dart', 'gbdt']
})

y_true, y_pred = list(), list()
X_test_for_SHAP = []
X_train, X_test = [], []
y_train, y_test = [], []
models = []
list_shap_values = list()
list_test_sets = list()
shap_values = None
sample_idx = 0
for train_ix, test_ix in cv.split(X):
    # split data
    X_train, X_test = X[train_ix, :], X[test_ix, :]
    y_train, y_test = y[train_ix], y[test_ix]
    X_test_for_SHAP.append(X_test)
    # print(X_test_for_SHAP)
    model = LGBMClassifier(random_state=42,
                           boosting_type='gbdt',
                           max_depth=4,
                           learning_rate=0.5,
                           reg_lambda=0.5,
                           min_data_in_leaf=8,
                           min_child_samples=2,
                           verbose=-1,
                           objective='binary'
                           )
    model.fit(X_train, y_train)
    # evaluate model
    ytemp = model.predict(X_test)
    # store
    models.append(model)
    y_true.append(y_test[0])
    y_pred.append(ytemp[0])
    # print(classification_report(y_true, y_pred, zero_division=1))
    scores = cross_val_score(model, X, y, scoring='accuracy', cv=cv, n_jobs=-1)
    accuracy = accuracy_score(y_true, y_pred)
    print("Accuracy: %.2f%%" % (accuracy * 100.0))
    print('Best score: {}'.format(scores))
    print("real value ", y_true)
    print("predicted ", y_pred)
    explainer = shap.Explainer(model, X)
    if shap_values is None:
        shap_values = explainer.shap_values(X_test, check_additivity=False)
    else:
        shap_values += explainer.shap_values(X_test, check_additivity=False)
    # list_shap_values.append(shap_values)
    list_test_sets.append(test_ix)

plot_conf_matrix(model, y_true, y_pred, classes)
plt.show()

dataframes = []
for i in range(len(models)):
    feat_importances = pandas.DataFrame(sorted(zip(models[i].feature_importances_, feature_cols_gender)), columns=['Value', 'Feature'])
    dataframes.append(feat_importances)

feat_importances_sum = reduce(lambda x,y: x.add(y, fill_value=0), dataframes)
print(feat_importances_sum)

sorted(zip(model.feature_importances_, feature_cols_gender), reverse=True)
sns_feat_imp = plt.figure(figsize=(20, 10))
sns.barplot(x="Value", y='Feature', data=feat_importances_sum.sort_values(by="Value", ascending=False))
plt.title('LightGBM Features (avg over folds)')
plt.show()

shap.initjs()
sample_idx = 0

for sample_idx in range(len(X_train)):
    explainer = shap.Explainer(models[sample_idx], X_train, feature_names=feature_cols_gender)
    shap_values = explainer.shap_values(X_train[sample_idx][:])
    print("Expected/Base Value: \n", explainer.expected_value)
    print("\n Shap Values for Sample %d: " % sample_idx, shap_values)
    print("\n")
    prediction = models[sample_idx].predict(X_train[sample_idx].reshape(1, -1))[0]
    print("Prediction From Model: ", prediction)
    print("Prediction From Adding SHAP Values to Base Value : ", explainer.expected_value + shap_values.sum())

    shap.decision_plot(explainer.expected_value, shap_values, X_train[[sample_idx]],
                       link='logit', feature_names=feature_cols_gender, show=False)
    file_name = f'{sample_idx} decision plot.png'
    plt.subplots_adjust(left=0.232)
    plt.title(f'Sample: {X_id[sample_idx]} Predicted: {prediction}, Real value: {y_true[sample_idx]}', y=1.07)
    plt.savefig(file_name)
    plt.show()
    shap_values_for_wf = explainer(X)
    file_name = f'{sample_idx} wf plot.png'
    shap.waterfall_plot(shap_values_for_wf[sample_idx], max_display=len(feature_cols_gender), show=False)
    plt.subplots_adjust(left=0.300)
    plt.subplots_adjust(bottom=0.14)
    plt.subplots_adjust(top=0.852)
    plt.title(f'Sample: {X_id[sample_idx]} \n Predicted: {prediction}, Real value: {y_true[sample_idx]}', x=-0.25,
              y=1.05)
    plt.savefig(file_name)
    plt.show()
    sample_idx += 1


cv = LeaveOneOut()
cv.get_n_splits(X)
y_true, y_pred = list(), list()
X_train, X_test = [], []
y_train, y_test = [], []

for train_ix, test_ix in cv.split(X_w):
    # split data
    X_train, X_test = X_w[train_ix, :], X_w[test_ix, :]
    y_train, y_test = y_w[train_ix], y_w[test_ix]
    model = LGBMClassifier(random_state=42, verbose=-1, min_data_leaf=2)
    model.fit(X_train, y_train)
    # evaluate model
    ytemp = model.predict(X_test)
    # store
    y_true.append(y_test[0])
    y_pred.append(ytemp[0])
    print(classification_report(y_true, y_pred, zero_division=1))
    scores = cross_val_score(model, X_w, y_w, scoring='accuracy', cv=cv, n_jobs=-1)
    accuracy = accuracy_score(y_true, y_pred)
    print("Accuracy: %.2f%%" % (accuracy * 100.0))
    print('Best score: {}'.format(scores))
    print("real value ", y_true)
    print("predicted ", y_pred)

_true, y_pred = list(), list()
X_train, X_test = [], []
y_train, y_test = [], []

GSCV_tuning_model(X_w, y_w, cv, parameters={
    'boosting_type': ['goss', 'dart', 'gbdt']
})

y_true, y_pred = list(), list()
X_test_for_SHAP = []
X_train, X_test = [], []
y_train, y_test = [], []
models = []
list_shap_values = list()
list_test_sets = list()
shap_values = None
sample_idx = 0
for train_ix, test_ix in cv.split(X_w):
    # split data
    X_train, X_test = X_w[train_ix, :], X_w[test_ix, :]
    y_train, y_test = y_w[train_ix], y_w[test_ix]
    X_test_for_SHAP.append(X_test)
    # print(X_test_for_SHAP)
    model = LGBMClassifier(random_state=42,
                           boosting_type='gbdt',
                           max_depth=2,
                           learning_rate=0.5,
                           min_data_in_leaf=8,
                           verbose=-1,
                           objective='binary'
                           )
    model.fit(X_train, y_train)
    # evaluate model
    ytemp = model.predict(X_test)
    # store
    models.append(model)
    y_true.append(y_test[0])
    y_pred.append(ytemp[0])
    # print(classification_report(y_true, y_pred, zero_division=1))
    scores = cross_val_score(model, X_w, y_w, scoring='accuracy', cv=cv, n_jobs=-1)
    accuracy = accuracy_score(y_true, y_pred)
    print("Accuracy: %.2f%%" % (accuracy * 100.0))
    print('Best score: {}'.format(scores))
    print("real value ", y_true)
    print("predicted ", y_pred)
"""    explainer = shap.Explainer(model, X_w)
    if shap_values is None:
        shap_values = explainer.shap_values(X_test, check_additivity=False)
    else:
        shap_values += explainer.shap_values(X_test, check_additivity=False)
    # list_shap_values.append(shap_values)
    list_test_sets.append(test_ix)"""

plot_conf_matrix(model, y_true, y_pred, classes)
plt.show()
lightgbm.plot_importance(model, ylabel=feature_cols_gender)
sorted(zip(model.feature_importances_, feature_cols_gender), reverse=True)
feature_imp = pandas.DataFrame(sorted(zip(model.feature_importances_, feature_cols_gender)), columns=['Value', 'Feature'])
sns_feat_imp = plt.figure(figsize=(20, 10))
sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value", ascending=False))
plt.title('LightGBM Features (avg over folds)')
plt.show()