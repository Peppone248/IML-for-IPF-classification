import pandas
import shap
import seaborn as sns
import xgboost
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBClassifier, plot_tree
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, StratifiedKFold, LeaveOneOut
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, f1_score, accuracy_score, classification_report
from useful_methods import plot_conf_matrix, features_encoding, GSCV_tuning_model, shap_graphs_lgbm

data = pandas.read_csv('new Data Set Fibrosi.csv')
df = pandas.DataFrame(data)

null_df = df.apply(lambda x: sum(x.isnull())).to_frame(name='count')
print(null_df)
# plt.plot(null_df.index, null_df['count'])

"""
***************** DATA RE-SHAPE *****************
"""

features_encoding(df)

# df = df.dropna(subset=['DLCO/VA'])

feature_cols_prognosis = ['2-DDCT KL-6', '2-DDCT MIR21', '2-DDCT MIR92A', 'Età', 'DLCO']

classes = ['LENTA', 'RAPIDA']

X = df[feature_cols_prognosis].values
y = df['RAPIDAVSLENTA'].values

cv = LeaveOneOut()
cv.get_n_splits(X)
y_true, y_pred = list(), list()
X_train, X_test = [], []
y_train, y_test = [], []

for train_ix, test_ix in cv.split(X):
    # split data
    X_train, X_test = X[train_ix, :], X[test_ix, :]
    y_train, y_test = y[train_ix], y[test_ix]
    model = XGBClassifier(random_state=42)
    model.fit(X_train, y_train)
    # evaluate model
    ytemp = model.predict(X_test)
    # store
    y_true.append(y_test[0])
    y_pred.append(ytemp[0])
    # print(classification_report(y_test, y_pred, zero_division=1))
    scores = cross_val_score(model, X, y, scoring='accuracy', cv=cv, n_jobs=-1)
    accuracy = accuracy_score(y_true, y_pred)
    print("Accuracy: %.2f%%" % (accuracy * 100.0))
    print('Best score: {}'.format(scores))
    # print('The best parameter for the XGB classifier are: {}'.format(model.get_params))
    print("real value ", y_true)
    print("predicted ", y_pred)

model.get_booster().feature_names = feature_cols_prognosis
xgboost.plot_importance(model.get_booster())
plt.show()
fig, ax = plt.subplots(figsize=(20, 20))
# plot_tree(model.get_booster(), num_trees=1, ax=ax)
# plt.show()


"""
In this second part we will tune the model through GridSearchCV, trying to found the best value
related to the best accuracy result.
This is the ranges on which we operate:
'max_depth': [2, 5, 8, 10, 13]
'min_child_weight': [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6]
'learning_rate': [0.05, 0.1, 0.2, 0,3 0.5, 1]
'reg_alpha': [x / 10 for x in range(0, 11)]
'reg_lambda': [x / 10 for x in range(0, 11)]
'gamma': [0, 1, 2, 3, 5, 8]
'max_delta_step': [0, 1, 2, 3, 5, 8]
'scale_pos_weight': [-0.5 + x / 10 for x in range(0, 11)]

"""
y_true, y_pred = list(), list()
X_train, X_test = [], []
y_train, y_test = [], []

GSCV_tuning_model(X, y, cv, parameters={
    'reg_alpha': [0.1, 0.2, 0.5, 0.7, 1],
    'learning_rate': [0.05, 0.1, 0.3, 0.5, 1],
    #'gamma': [0, 1, 2, 3, 4]
    'booster': ['gbtree', 'gblinear', 'dart'],
    #'max_depth': [2, 4, 5, 8]
})

y_true, y_pred = list(), list()
X_train, X_test = [], []
y_train, y_test = [], []

for train_ix, test_ix in cv.split(X):
    # split data
    X_train, X_test = X[train_ix, :], X[test_ix, :]
    y_train, y_test = y[train_ix], y[test_ix]
    model_tuned = XGBClassifier(random_state=42,
                                booster='gbtree',
                                reg_alpha=0.1,
                                reg_lambda=0.2,
                                learning_rate=1,
                                max_depth=2,
                                #gamma=3
                                #n_estimators=20,
                                #gamma=3
                                )
    model_tuned.fit(X_train, y_train)
    # evaluate model
    ytemp = model_tuned.predict(X_test)
    # store
    y_true.append(y_test[0])
    y_pred.append(ytemp[0])
    # print(classification_report(y_test, y_pred, zero_division=1))
    scores = cross_val_score(model_tuned, X, y, scoring='accuracy', cv=cv, n_jobs=-1)
    accuracy = accuracy_score(y_true, y_pred)
    print("Accuracy: %.2f%%" % (accuracy * 100.0))
    print('Best score: {}'.format(scores))
    # print('The best parameter for the XGB classifier are: {}'.format(model.get_params))
    print("real value ", y_true)
    print("predicted ", y_pred)

model_tuned.get_booster().feature_names = feature_cols_prognosis

plot_conf_matrix(model_tuned, y_true, y_pred, classes)

count = 0
xgboost.plot_importance(model_tuned.get_booster(), importance_type='gain')
xgboost.plot_importance(model_tuned.get_booster(), importance_type='weight')
fig, ax = plt.subplots(figsize=(20, 20))
for i in range(len(feature_cols_prognosis)):
    plot_tree(model_tuned.get_booster(), num_trees=count, ax=ax)
    plt.show()
    count += 1
# calculate accuracy
# print(cv.get_n_splits(X))

"""shap.initjs()
shap_global_interpret_graphs(model, X, X_train, feature_cols_prognosis)
sample_idx = 14
print(X_train[sample_idx][:])
for i in range(len(X_train)):
    explainer = shap.Explainer(model, X_train, feature_names=feature_cols_prognosis)
    shap_values = explainer.shap_values(X_train[sample_idx][:])
    print("Expected/Base Value: \n", explainer.expected_value)
    print("\n Shap Values for Sample %d: " % sample_idx, shap_values)
    print("\n")
    prediction = model.predict(X_train[sample_idx].reshape(1, -1))[0]
    print("Prediction From Model: ", prediction)
    print("Prediction From Adding SHAP Values to Base Value : ", explainer.expected_value + shap_values.sum())

    shap.decision_plot(explainer.expected_value, shap_values, X_train[[sample_idx]],
                       link='logit', feature_names=feature_cols_prognosis, show=False)
    file_name = f'{sample_idx} decision plot.png'
    plt.subplots_adjust(left=0.252)
    plt.title(f'Predicted: {prediction}, Real value: {y_true[sample_idx]}', y=1.05)
    plt.savefig(file_name)
    plt.show()
    shap_values_for_wf = explainer(X)
    file_name = f'{sample_idx} wf plot.png'
    shap.waterfall_plot(shap_values_for_wf[sample_idx], max_display=len(feature_cols_prognosis), show=False)
    plt.subplots_adjust(left=0.300)
    plt.title(f'Predicted: {prediction}, Real value: {y_true[sample_idx]}', y=1.05)
    plt.savefig(file_name)
    plt.show()
    sample_idx += 1"""
