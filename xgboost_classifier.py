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
from grid_search_utils import plot_grid_search, table_grid_search
from useful_methods import plot_conf_matrix, features_encoding, GSCV_tuning_model, shap_global_charts
from functools import reduce
import dice_ml

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

feature_cols = ['Fumo', 'FVC%', 'FEV1%', 'DLCO', 'Macro%', 'Neu%', 'Lin%',
                'Età', 'Genere', '2-DDCT KL-6', '2-DDCT MIR21', 'FEV1%/FVC%', '2-DDCT MIR92A']

classes = ['ALTRO', 'IPF']

X_not_converted = df[feature_cols]
X = df[feature_cols].values
y = df['IPFVSALTRO'].values
X_id = df['ID Lab'].values

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

model.get_booster().feature_names = feature_cols
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
    'reg_alpha': [x / 10 for x in range(0, 11)],
    'max_depth': [2, 4, 5, 8]
})

y_true, y_pred = list(), list()
X_train, X_test = [], []
y_train, y_test = [], []
models = []
list_shap_values = list()
list_test_sets = list()
shap_values = None

for train_ix, test_ix in cv.split(X):
    # split data
    X_train, X_test = X[train_ix, :], X[test_ix, :]
    y_train, y_test = y[train_ix], y[test_ix]
    model = XGBClassifier(random_state=42,
                          max_depth=5,
                          learning_rate=0.05,
                          n_estimators=20,
                          gamma=3
                          )
    model.fit(X_train, y_train)
    # evaluate model
    ytemp = model.predict(X_test)
    models.append(model)
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
    explainer = shap.TreeExplainer(model)
    if shap_values is None:
        shap_values = explainer.shap_values(X_test)
    else:
        shap_values += explainer.shap_values(X_test)
    # list_shap_values.append(shap_values)
    list_test_sets.append(test_ix)

X_train_df = pandas.DataFrame(X_train,
                              columns=feature_cols)
df_class = df['IPFVSALTRO'].iloc[:-1]
X_train_df['IPFVSALTRO'] = df_class.values

test_set = list_test_sets[0]
for i in range(2, len(list_test_sets)):
    test_set = np.concatenate((test_set, list_test_sets[i]), axis=0)

# bringing back variable names
X_test = pandas.DataFrame(X[test_set], columns=feature_cols)
print('X_test: ', X_test)

print(type(shap_values))
print(shap_values)
shap_values /= 38
print('shap v after rounding: ', shap_values)

model.get_booster().feature_names = feature_cols
plot_conf_matrix(model, y_true, y_pred, classes)

count = 0
dataframes = []
for i in range(len(models)):
    feat_importances = pandas.DataFrame(models[i].feature_importances_, index=X_not_converted.columns, columns=["Importance"])
    dataframes.append(feat_importances)

feat_importances_sum = reduce(lambda x,y: x.add(y, fill_value=0), dataframes)
print(feat_importances_sum)

feat_importances_sum.sort_values(by='Importance', ascending=False, inplace=True)
feat_importances_sum.plot(kind='bar', figsize=(9, 7))
plt.show()

"""
for i in range(len(models)):
    xgboost.plot_importance(models[i].get_booster(), importance_type='gain')
    xgboost.plot_importance(models[i].get_booster(), importance_type='weight')
"""

fig, ax = plt.subplots(figsize=(20, 20))
for i in range(len(feature_cols)):
    plot_tree(model.get_booster(), num_trees=count, ax=ax)
    plt.show()
    count += 1
# calculate accuracy
# print(cv.get_n_splits(X))

dice_data = dice_ml.Data(dataframe=X_train_df, continuous_features=feature_cols, outcome_name='IPFVSALTRO')

dice_model = dice_ml.Model(model=model, backend='sklearn')
exp = dice_ml.Dice(dice_data, dice_model, method='random')
e = exp.generate_counterfactuals(X_test, total_CFs=5, desired_class="opposite", features_to_vary=['Genere', 'FVC%', 'Lin%', '2-DDCT KL-6', 'FEV1%', 'DLCO'])

e.cf_examples_list[18].visualize_as_dataframe(show_only_changes=True, display_sparse_df=False)
e.cf_examples_list[19].visualize_as_dataframe(show_only_changes=True, display_sparse_df=False)


shap.summary_plot(shap_values, plot_type='bar', feature_names=feature_cols, show=True)
shap_global_charts(models, 0, X_test, feature_cols)

shap.initjs()
sample_idx = 0
for sample_idx in range(len(X_train)):
    explainer = shap.Explainer(models[sample_idx], X_train, feature_names=feature_cols)
    shap_values = explainer.shap_values(X_train[sample_idx][:])
    print("Expected/Base Value: \n", explainer.expected_value)
    print("\n Shap Values for Sample %d: " % sample_idx, shap_values)
    print("\n")
    prediction = model.predict(X_train[sample_idx].reshape(1, -1))[0]
    print("Prediction From Model: ", prediction)
    print("Prediction From Adding SHAP Values to Base Value : ", explainer.expected_value + shap_values.sum())

    shap.decision_plot(explainer.expected_value, shap_values, X_train[[sample_idx]],
                       link='logit', feature_names=feature_cols, show=False)
    file_name = f'{sample_idx} decision plot.png'
    plt.subplots_adjust(left=0.252)
    plt.title(f'Sample: {X_id[sample_idx]} \n Predicted: {prediction}, Real value: {y_true[sample_idx]}', y=1.05)
    plt.savefig(file_name)
    plt.show()
    shap_values_for_wf = explainer(X)
    file_name = f'{sample_idx} wf plot.png'
    shap.waterfall_plot(shap_values_for_wf[sample_idx], max_display=len(feature_cols), show=False)
    plt.subplots_adjust(left=0.300)
    plt.title(f'Sample: {X_id[sample_idx]} \n Predicted: {prediction}, Real value: {y_true[sample_idx]}', y=1.05)
    plt.savefig(file_name)
    plt.show()
    sample_idx += 1
