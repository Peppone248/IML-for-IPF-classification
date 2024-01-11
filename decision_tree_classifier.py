import dice_ml
import numpy as np
import pandas
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from sklearn.model_selection import train_test_split, GridSearchCV, LeaveOneOut, cross_val_score
from sklearn.tree import _tree
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score, precision_recall_curve
from useful_methods import features_encoding, shap_graphs_decision_tree, plot_conf_matrix, GSCV_tuning_model, \
    shap_global_charts_tree_exp
from functools import reduce
import scipy.spatial as sp
import scipy

pandas.set_option('display.max_columns', None)
data = pandas.read_csv('new Data Set Fibrosi.csv')
df = pandas.DataFrame(data)

features_encoding(df)

# df = df.drop('ID Lab', axis=1)
df = df.drop('Patologia', axis=1)

df = df.dropna()

feature_cols = ['Genere', 'Fumo', 'FVC%', 'FEV1%', 'DLCO', 'Macro%', 'Neu%', 'Lin%', 'Età',
                '2-DDCT KL-6', '2-DDCT MIR21', 'FEV1%/FVC%', '2-DDCT MIR92A']

"""
COSINE SIMILARITY AND SCALAR PRODUCT MATRICES
"""

cosine_sim_feat_by_rows = df
cosine_sim_feat_by_rows = cosine_sim_feat_by_rows.set_index('ID Lab')
print(cosine_sim_feat_by_rows)
v = cosine_similarity(cosine_sim_feat_by_rows.values)
cosine_sim_feat_by_rows = pandas.DataFrame(v, columns=df['ID Lab'])
cosine_sim_feat_by_rows.index = df['ID Lab']
print(cosine_sim_feat_by_rows)

sns.set()
sns.heatmap(cosine_sim_feat_by_rows, annot=True, cmap='Blues', fmt='.1f')
plt.legend([],[], frameon=False)
plt.show()

new_df = df[feature_cols]

scalar_product_feat_df = pandas.DataFrame()
for i in range(len(feature_cols)):
    scalar_prod_features = df[feature_cols[i]].dot(new_df)
    scalar_product_feat_df[i]=scalar_prod_features
    print('Scalar product of ', str(feature_cols[i]),': \n', scalar_prod_features)

scalar_product_feat_df.columns = feature_cols
print(scalar_product_feat_df)
sns.set()
sns.heatmap(scalar_product_feat_df, annot=True, cmap='Blues', fmt='.1f')
plt.legend([],[], frameon=False)
plt.show()

"""
Features cosine matrix by columns
"""

cosine_result = 1 - sp.distance.cdist(new_df.T, new_df.T, 'cosine')
# print(str(feature_cols), ': \n', cosine_result, '\n')
cosine_df = pandas.DataFrame(cosine_result)
cosine_df.columns = feature_cols
cosine_df.index = feature_cols
print(cosine_df)
cosine_df.to_csv('cosine similarity features.csv')

sns.set()
#cosine_df = cosine_df.set_index('Genere')
sns.heatmap(cosine_df, annot=True, fmt='.2f', linewidths=.7)
plt.legend([],[], frameon=False)
plt.show()

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

plt.figure(figsize=(10, 10))
plot_tree(model, feature_names=feature_cols, class_names=classes, max_depth=model.max_depth, filled=True)

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

test_set = list_test_sets[0]
shap_values = np.array(list_shap_values[0])
for i in range(2, len(list_test_sets)):
    test_set = np.concatenate((test_set, list_test_sets[i]), axis=0)
    shap_values = np.concatenate((shap_values, np.array(list_shap_values[i])), axis=1)

# bringing back variable names
X_test = pandas.DataFrame(X[test_set], columns=feature_cols)

""" Cosine similarity and scalar product matrices for shap values """

list_sv_for_cosine = []
for i in range(len(list_shap_values)):
    list_sv_for_cosine.append(list_shap_values[i][1][0])
    print(list_shap_values[i][1][0], end=" ")
print()
# print(list_sv_for_cosine)

sv_first_istance = list_sv_for_cosine[26]
sv_second_istance = list_sv_for_cosine[5]

similarity = cosine_similarity([sv_first_istance], [sv_second_istance])[0][0]
print(similarity)

scalar_prod_sv = np.dot(list_sv_for_cosine, np.array(list_sv_for_cosine).T.tolist())
print(scalar_prod_sv)

scalar_prod_sv_df = pandas.DataFrame(scalar_prod_sv)
scalar_prod_sv_df.columns = df['ID Lab']
scalar_prod_sv_df.index = df['ID Lab']
print(scalar_prod_sv_df)
sns.set()
scalar_prod_sv_df = scalar_prod_sv_df.set_index(df['ID Lab'])
sns.heatmap(scalar_prod_sv_df, annot=True, fmt='.2f', linewidths=.2)
plt.legend([],[], frameon=False)
plt.show()

cosine_sv = 1 - sp.distance.cdist(list_sv_for_cosine, list_sv_for_cosine, 'cosine')
cosine_sim_sv_df = pandas.DataFrame(cosine_sv)
cosine_sim_sv_df.columns = df['ID Lab']
cosine_sim_sv_df.index = df['ID Lab']
print(cosine_sim_sv_df)
sns.set()
cosine_sim_sv_df = cosine_sim_sv_df.set_index(df['ID Lab'])
sns.heatmap(cosine_sim_sv_df, annot=True, fmt='.2f', linewidths=.7)
plt.legend([],[], frameon=False)
plt.show()


for i in range(len(feature_cols)):
    scalar_prod_features = df[feature_cols[i]].dot(new_df)
    scalar_product_feat_df[i] = scalar_prod_features
    print('Scalar product of ', str(feature_cols[i]),': \n', scalar_prod_features)

# scalar_product_feat_df.columns = feature_cols
print(scalar_product_feat_df)
sns.set()
sns.heatmap(scalar_product_feat_df, annot=True, cmap='Blues', fmt='.1f')
plt.legend([],[], frameon=False)
plt.show()


""" Counterfactuals calculation """

"""for i in range(len(list_shap_values)): """

"""dice_data = dice_ml.Data(dataframe=X_train_df, continuous_features=continuous_features, outcome_name='IPFVSALTRO')
dice_model = dice_ml.Model(model=model, backend='sklearn')
exp = dice_ml.Dice(dice_data, dice_model, method='random')
e = exp.generate_counterfactuals(X_test, total_CFs=5, desired_class="opposite")
e.cf_examples_list[1].final_cfs_df.to_csv(path_or_buf='counterfactuals_dt - 1.csv', index=True)
e.cf_examples_list[7].final_cfs_df.to_csv(path_or_buf='counterfactuals_dt - 7.csv', index=True)
e.cf_examples_list[8].final_cfs_df.to_csv(path_or_buf='counterfactuals_dt - 8.csv', index=True)
e.cf_examples_list[12].final_cfs_df.to_csv(path_or_buf='counterfactuals_dt - 12.csv', index=True)
e.cf_examples_list[13].final_cfs_df.to_csv(path_or_buf='counterfactuals_dt - 13.csv', index=True)
e.visualize_as_dataframe(show_only_changes=True)"""

""" Plot SHAP charts """


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




