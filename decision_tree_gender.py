import dice_ml
import numpy as np
import pandas
import matplotlib.pyplot as plt
import shap
from sklearn.model_selection import train_test_split, GridSearchCV, LeaveOneOut, cross_val_score
from sklearn.tree import _tree
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score, precision_recall_curve
from useful_methods import features_encoding, shap_graphs_decision_tree, plot_conf_matrix, GSCV_tuning_model, \
    shap_global_charts_tree_exp
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
df_men = df_men.dropna()

feature_cols_gender = ['2-DDCT KL-6', '2-DDCT MIR21', '2-DDCT MIR92A', 'Macro%', 'Neu%']

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

""" Classification task for men's dataset """

for train_ix, test_ix in cv.split(X):
    # split data
    X_train, X_test = X[train_ix, :], X[test_ix, :]
    y_train, y_test = y[train_ix], y[test_ix]
    model = DecisionTreeClassifier(criterion='gini',
                                   random_state=42)
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
    print("real value ", y_true)
    print("predicted ", y_pred)

plt.figure(figsize=(10, 10))
plot_tree(model, feature_names=feature_cols_gender, class_names=['ALTRO', 'IPF'], max_depth=model.max_depth,
          filled=True)

plt.show()

y_true, y_pred = list(), list()
X_train, X_test = [], []
y_train, y_test = [], []

GSCV_tuning_model(X, y, cv, parameters={
    'criterion': ['gini', 'entropy'],
    'splitter': ['random', 'best'],
    'max_depth': [2, 4, 6, 8, 10],
    'min_samples_split': [2, 4, 6, 8, 10]
})

y_true, y_pred = list(), list()
X_train, X_test = [], []
y_train, y_test = [], []
models = []
shap_values = None
list_test_sets = list()
list_shap_values = list()

for train_ix, test_ix in cv.split(X):
    # split data
    X_train, X_test = X[train_ix, :], X[test_ix, :]
    y_train, y_test = y[train_ix], y[test_ix]
    model = DecisionTreeClassifier(criterion='entropy',
                                   splitter='random',
                                   random_state=42,
                                   max_depth=8,
                                   min_samples_split=10
                                   )
    model.fit(X_train, y_train)
    # evaluate model
    ytemp = model.predict(X_test)
    models.append(model)
    y_prob_train = model.predict_proba(X_train)[:, 1]
    y_prob_test = model.predict_proba(X_test)[:, 1]
    # store
    y_true.append(y_test[0])
    y_pred.append(ytemp[0])
    # print(classification_report(y_test, y_pred, zero_division=1))
    scores = cross_val_score(model, X, y, scoring='accuracy', cv=cv, n_jobs=-1)
    accuracy = accuracy_score(y_true, y_pred)
    print("Accuracy: %.2f%%" % (accuracy * 100.0))
    print('Best score: {}'.format(scores))
    print("real value ", y_true)
    print("predicted ", y_pred)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)
    list_shap_values.append(shap_values)
    list_test_sets.append(test_ix)

"""X_train_df = pandas.DataFrame(X_train,
                              columns=feature_cols_gender)
df_class = df['IPFVSALTRO'].iloc[:-1]
X_train_df['IPFVSALTRO'] = df_class.values"""

precision, recall, thresholds = precision_recall_curve(y_train, y_prob_train)
plt.fill_between(recall, precision)
plt.ylabel("Precision")
plt.xlabel("Recall")
plt.title("Train Precision-Recall curve")
plt.show()

plt.figure(figsize=(10, 10))
plot_tree(model, feature_names=feature_cols_gender, class_names=classes, max_depth=model.max_depth, filled=True)
plt.show()

plot_conf_matrix(model, y_true, y_pred, classes)

dataframes = []
for i in range(len(models)):
    feat_importances = pandas.DataFrame(models[i].feature_importances_, index=X_not_converted.columns, columns=["Importance"])
    dataframes.append(feat_importances)

feat_importances_sum = reduce(lambda x,y: x.add(y, fill_value=0), dataframes)
print(feat_importances_sum)


feat_importances_sum.sort_values(by='Importance', ascending=False, inplace=True)
feat_importances_sum.plot(kind='bar', figsize=(9, 7))
plt.show()

shap.initjs()
sample_idx = 0
# shap.summary_plot(shap_values[1][:], plot_type='bar', feature_names=feature_cols, show=False)
plt.show()
#shap_global_charts_tree_exp(models, sample_idx, X_test, feature_cols_gender)
for sample_idx in range(len(X_train)):
    shap_graphs_decision_tree(models[sample_idx], X, X_train, feature_cols_gender, sample_idx, y_true, X_id)
    sample_idx += 1

y_true, y_pred = list(), list()
X_train, X_test = [], []
y_train, y_test = [], []
""" Classification task for women's dataset """

for train_ix, test_ix in cv.split(X_w):
    # split data
    X_train, X_test = X_w[train_ix, :], X_w[test_ix, :]
    y_train, y_test = y_w[train_ix], y_w[test_ix]
    model = DecisionTreeClassifier(criterion='gini',
                                   random_state=42)
    model.fit(X_train, y_train)
    # evaluate model
    ytemp = model.predict(X_test)
    # store
    y_true.append(y_test[0])
    y_pred.append(ytemp[0])
    # print(classification_report(y_test, y_pred, zero_division=1))
    scores = cross_val_score(model, X_w, y_w, scoring='accuracy', cv=cv, n_jobs=-1)
    accuracy = accuracy_score(y_true, y_pred)
    print("Accuracy: %.2f%%" % (accuracy * 100.0))
    print('Best score: {}'.format(scores))
    print("real value ", y_true)
    print("predicted ", y_pred)

plt.figure(figsize=(10, 10))
plot_tree(model, feature_names=feature_cols_gender, class_names=['ALTRO', 'IPF'], max_depth=model.max_depth,
          filled=True)

plt.show()

y_true, y_pred = list(), list()
X_train, X_test = [], []
y_train, y_test = [], []

GSCV_tuning_model(X_w, y_w, cv, parameters={
    'criterion': ['gini', 'entropy'],
    'splitter': ['random', 'best'],
    'max_depth': [2, 4, 6, 8, 10],
    'min_samples_split': [2, 4, 6, 8, 10]
})

y_true, y_pred = list(), list()
X_train, X_test = [], []
y_train, y_test = [], []
models = []
shap_values = None
list_test_sets = list()
list_shap_values = list()

for train_ix, test_ix in cv.split(X_w):
    # split data
    X_train, X_test = X_w[train_ix, :], X_w[test_ix, :]
    y_train, y_test = y_w[train_ix], y_w[test_ix]
    model = DecisionTreeClassifier(criterion='entropy',
                                   splitter='best',
                                   random_state=42,
                                   max_depth=8,
                                   )
    model.fit(X_train, y_train)
    # evaluate model
    ytemp = model.predict(X_test)
    models.append(model)
    y_prob_train = model.predict_proba(X_train)[:, 1]
    y_prob_test = model.predict_proba(X_test)[:, 1]
    # store
    y_true.append(y_test[0])
    y_pred.append(ytemp[0])
    # print(classification_report(y_test, y_pred, zero_division=1))
    scores = cross_val_score(model, X_w, y_w, scoring='accuracy', cv=cv, n_jobs=-1)
    accuracy = accuracy_score(y_true, y_pred)
    print("Accuracy: %.2f%%" % (accuracy * 100.0))
    print('Best score: {}'.format(scores))
    print("real value ", y_true)
    print("predicted ", y_pred)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)
    list_shap_values.append(shap_values)
    list_test_sets.append(test_ix)

plt.figure(figsize=(10, 10))
plot_tree(model, feature_names=feature_cols_gender, class_names=classes, max_depth=model.max_depth, filled=True)
plt.show()

plot_conf_matrix(model, y_true, y_pred, classes)

feat_importances = pandas.DataFrame(model.feature_importances_, index=X_not_converted.columns, columns=["Importance"])
feat_importances.sort_values(by='Importance', ascending=False, inplace=True)
feat_importances.plot(kind='bar', figsize=(9, 7))
plt.show()
