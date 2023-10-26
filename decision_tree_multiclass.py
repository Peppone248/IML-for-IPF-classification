import numpy as np
import pandas
import matplotlib.pyplot as plt
import shap
from shap import Explainer, Explanation
from sklearn.model_selection import GridSearchCV, LeaveOneOut, cross_val_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import _tree
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score
import scikitplot as skplot
from grid_search_utils import plot_grid_search, table_grid_search
from useful_methods import features_encoding, shap_graphs_decision_tree, plot_conf_matrix, GSCV_tuning_model

pandas.set_option('display.max_columns', None)
data = pandas.read_csv('new Data Set Fibrosi.csv')
df = pandas.DataFrame(data)

features_encoding(df)

df = df.drop('ID Lab', axis=1)

df = df.dropna()

feature_cols = ['Età', 'Fumo', 'FVC%', 'FEV1%', 'DLCO', 'Macro%', 'Neu%', 'Lin%', '2-DDCT KL-6', '2-DDCT MIR21',
                '2-DDCT MIR92A']

classes = ['ALTRO', 'IPF', 'HP', 'NSIP']

X_not_converted = df[feature_cols]
X = df[feature_cols].values
y = df['Patologia'].values

"""scaler = MinMaxScaler(feature_range=(0, 1))
scaler.fit(X)
X = scaler.transform(df[feature_cols].values)"""

cv = LeaveOneOut()
cv.get_n_splits(X)
y_true, y_pred = list(), list()
X_train, X_test = [], []
y_train, y_test = [], []

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
plot_tree(model, feature_names=feature_cols, class_names=['ALTRO', 'IPF', 'HP', 'NSIP'], max_depth=model.max_depth,
          filled=True)

plot_conf_matrix(model, y_true, y_pred, classes)

feat_importances = pandas.DataFrame(model.feature_importances_, index=X_not_converted.columns, columns=["Importance"])
feat_importances.sort_values(by='Importance', ascending=False, inplace=True)
feat_importances.plot(kind='bar', figsize=(9, 7))
plt.show()

"""
In this second part we will tune the model through GridSearchCV, trying to found the best value
related to the best accuracy result.
This is the ranges on which we operate:
'criterion': ['gini', 'entropy'],
'splitter': ['best', 'random'],
'min_samples_split': [2, 5, 8, 13, 21],
'min_samples_leaf': [2, 5, 8, 13, 21],
'max_features': ['auto', 'sqrt', 'log2', 2, 5, 8, 10, 13]    
'max_depth': [2, 5, 8, 10, 13]
"""

GSCV_tuning_model(X, y, cv, parameters={'criterion': ['gini', 'entropy'],
                                        'max_depth': np.arange(1, 21).tolist()[0::2],
                                        #'min_samples_split': np.arange(2, 11).tolist()[0::2],
                                        'max_leaf_nodes':np.arange(3,26).tolist()[0::2]
                                        })

y_true, y_pred = list(), list()
X_train, X_test = [], []
y_train, y_test = [], []
for train_ix, test_ix in cv.split(X):
    # split data
    X_train, X_test = X[train_ix, :], X[test_ix, :]
    y_train, y_test = y[train_ix], y[test_ix]
    model = DecisionTreeClassifier(criterion='gini',
                                   splitter='random',
                                   min_samples_split=4,
                                   max_leaf_nodes=21,
                                   random_state=42,
                                   max_depth=1,
                                   )
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
plot_tree(model, feature_names=feature_cols, class_names=['ALTRO', 'IPF', 'HP', 'NSIP'], max_depth=model.max_depth,
          filled=True)

plot_conf_matrix(model, y_true, y_pred, classes)

feat_importances = pandas.DataFrame(model.feature_importances_, index=X_not_converted.columns, columns=["Importance"])
feat_importances.sort_values(by='Importance', ascending=False, inplace=True)
feat_importances.plot(kind='bar', figsize=(9, 7))
plt.show()

shap.initjs()
sample_idx = 1
# shap_graphs_decision_tree(model, X_train, feature_cols, sample_idx, y_pred)

"""def get_rules(tree, feature_names, class_names):
    tree_ = tree.tree_
    feature_name = [
        feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
        for i in tree_.feature
    ]

    paths = []
    path = []

    def recurse(node, path, paths):

        if tree_.feature[node] != _tree.TREE_UNDEFINED:
            name = feature_name[node]
            threshold = tree_.threshold[node]
            p1, p2 = list(path), list(path)
            p1 += [f"({name} <= {np.round(threshold, 0)})"]
            recurse(tree_.children_left[node], p1, paths)
            p2 += [f"({name} > {np.round(threshold, 3)})"]
            recurse(tree_.children_right[node], p2, paths)
        else:
            path += [(tree_.value[node], tree_.n_node_samples[node])]
            paths += [path]

    recurse(0, path, paths)

    # sort by samples count
    samples_count = [p[-1][1] for p in paths]
    ii = list(np.argsort(samples_count))
    paths = [paths[i] for i in reversed(ii)]

    rules = []
    for path in paths:
        rule = "if "

        for p in path[:-1]:
            if rule != "if ":
                rule += " and "
            rule += str(p)
        rule += " then "
        if class_names is None:
            rule += "response: " + str(np.round(path[-1][0][0][0], 3))
        else:
            classes = path[-1][0][0]
            l = np.argmax(classes)
            rule += f"class: {class_names[l]} (proba: {np.round(100.0 * classes[l] / np.sum(classes), 2)}%)"
        rule += f" | based on {path[-1][1]:,} samples"
        rules += [rule]

    return rules


rules = get_rules(model, feature_names=feature_cols, class_names=['ALTRO', 'IPF'])
for r in rules: print(r)
"""
