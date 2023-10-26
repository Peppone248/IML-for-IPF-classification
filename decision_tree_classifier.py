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

pandas.set_option('display.max_columns', None)
data = pandas.read_csv('new Data Set Fibrosi.csv')
df = pandas.DataFrame(data)

# print(df_women)

features_encoding(df)


# df = df.drop('ID Lab', axis=1)
df = df.drop('Patologia', axis=1)

# df = df.dropna()

feature_cols = ['Genere', 'Fumo', 'FVC%', 'FEV1%', 'DLCO', 'Macro%', 'Neu%', 'Lin%', 'Età',
                '2-DDCT KL-6', '2-DDCT MIR21', 'FEV1%/FVC%', '2-DDCT MIR92A']

classes = ['ALTRO', 'IPF']

continuous_features = ['FVC%', 'FEV1%', 'DLCO', 'Macro%', 'Neu%', 'Lin%', 'Età',
                       '2-DDCT KL-6', '2-DDCT MIR21', 'FEV1%/FVC%', '2-DDCT MIR92A']

X_not_converted = df[feature_cols]
X = df[feature_cols].values
y = df['IPFVSALTRO'].values
X_id = df['ID Lab'].values

unique, counts = np.unique(y, return_counts=True)
plt.pie(counts, labels=classes, autopct='%.0f%%')

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
plot_tree(model, feature_names=feature_cols, class_names=['ALTRO', 'IPF'], max_depth=model.max_depth, filled=True)

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
y_true, y_pred = list(), list()
X_train, X_test = [], []
y_train, y_test = [], []

GSCV_tuning_model(X, y, cv, parameters={
    'criterion': ['gini', 'entropy'],
    'splitter': ['random', 'best'],
    'min_samples_split': [2, 4, 6, 8, 10],
    'min_samples_leaf': [1, 2, 4, 5, 8, 10]
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
    model = DecisionTreeClassifier(criterion='gini',
                                   splitter='random',
                                   min_samples_leaf=4,
                                   min_samples_split=2,
                                   random_state=42,
                                   max_depth=2
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


X_train_df = pandas.DataFrame(X_train,
                              columns=['Genere', 'Fumo', 'FVC%', 'FEV1%', 'DLCO', 'Macro%', 'Neu%', 'Lin%', 'Età',
                                       '2-DDCT KL-6', '2-DDCT MIR21', 'FEV1%/FVC%', '2-DDCT MIR92A'])
df_class = df['IPFVSALTRO'].iloc[:-1]
X_train_df['IPFVSALTRO'] = df_class.values

precision, recall, thresholds = precision_recall_curve(y_train, y_prob_train)
plt.fill_between(recall, precision)
plt.ylabel("Precision")
plt.xlabel("Recall")
plt.title("Train Precision-Recall curve")
plt.show()

plt.figure(figsize=(10, 10))
plot_tree(model, feature_names=feature_cols, class_names=classes, max_depth=model.max_depth, filled=True)

plot_conf_matrix(model, y_true, y_pred, classes)

feat_importances = pandas.DataFrame(model.feature_importances_, index=X_not_converted.columns, columns=["Importance"])
feat_importances.sort_values(by='Importance', ascending=False, inplace=True)
feat_importances.plot(kind='bar', figsize=(9, 7))

test_set = list_test_sets[0]
shap_values = np.array(list_shap_values[0])
for i in range(2, len(list_test_sets)):
    test_set = np.concatenate((test_set, list_test_sets[i]), axis=0)
    shap_values = np.concatenate((shap_values, np.array(list_shap_values[i])), axis=1)

# bringing back variable names
X_test = pandas.DataFrame(X[test_set], columns=feature_cols)
print('X_test: ', X_test)
print(X_test.dtypes)
# print(shap_values[1][:])

dice_data = dice_ml.Data(dataframe=X_train_df, continuous_features=continuous_features, outcome_name='IPFVSALTRO')
dice_model = dice_ml.Model(model=model, backend='sklearn')
exp = dice_ml.Dice(dice_data, dice_model, method='random')
e = exp.generate_counterfactuals(X_test, total_CFs=5, desired_class="opposite")
e.cf_examples_list[1].final_cfs_df.to_csv(path_or_buf='counterfactuals_dt - 1.csv', index=True)
e.cf_examples_list[7].final_cfs_df.to_csv(path_or_buf='counterfactuals_dt - 7.csv', index=True)
e.cf_examples_list[8].final_cfs_df.to_csv(path_or_buf='counterfactuals_dt - 8.csv', index=True)
e.cf_examples_list[12].final_cfs_df.to_csv(path_or_buf='counterfactuals_dt - 12.csv', index=True)
e.cf_examples_list[13].final_cfs_df.to_csv(path_or_buf='counterfactuals_dt - 13.csv', index=True)
e.visualize_as_dataframe(show_only_changes=True)

shap.initjs()
sample_idx = 0
# shap.summary_plot(shap_values[1][:], plot_type='bar', feature_names=feature_cols, show=False)
plt.show()
shap_global_charts_tree_exp(models, sample_idx, X_test, feature_cols)
for sample_idx in range(len(X_train)):
    shap_graphs_decision_tree(models[sample_idx], X, X_train, feature_cols, sample_idx, y_true, X_id)
    sample_idx += 1


def get_rules(tree, feature_names, class_names):
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
            p1 += [f"({name} <= {np.round(threshold, 1)})"]
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