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

features_encoding(df)

df = df.drop('Patologia', axis=1)

feature_cols = ['Genere', 'FVC%', 'FEV1%', 'DLCO', 'Macro%', 'Neu%', 'Lin%', '2-DDCT KL-6', '2-DDCT MIR21']
classes = ['ALTRO', 'IPF']
continuous_features = ['FVC%', 'FEV1%', 'DLCO', 'Macro%', 'Neu%', 'Lin%',
                       '2-DDCT KL-6', '2-DDCT MIR21']

# X_not_converted = df[feature_cols]
X_display = df[feature_cols]
X = df[feature_cols].values
y = df['IPFVSALTRO'].values
X_id = df['ID Lab'].values

unique, counts = np.unique(y, return_counts=True)
plt.pie(counts, labels=classes, autopct=lambda p: '{:.2f}%  ({:,.0f})'.format(p, p * sum(counts) / 100))
plt.show()

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
                           verbose=-1,
                           objective='binary',
                           metric='auc',
                           subsample=0.7
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

X_train_df = pandas.DataFrame(X_train,
                              columns=feature_cols)
df_class = df['IPFVSALTRO'].iloc[:-1]
X_train_df['IPFVSALTRO'] = df_class.values

test_set = list_test_sets[0]
for i in range(1, len(list_test_sets)):
    test_set = np.concatenate((test_set, list_test_sets[i]), axis=0)

# bringing back variable names
X_test = pandas.DataFrame(X[test_set], columns=feature_cols)
# X_test['Genere'] = pd.to_numeric(X_test['Genere'])
print('X_test: ', X_test)
print(X_test.dtypes)

plot_conf_matrix(model, y_true, y_pred, classes)
plt.show()
# lightgbm.plot_importance(model, importance_type='gain',ylabel=['Genere', 'FVC%', 'FEV1%', 'DLCO', 'DLCO/VA', 'Macro%', 'Neu%', 'Lin%', '2-DDCT KL-6', '2-DDCT MIR21'])

sorted(zip(model.feature_importances_, feature_cols), reverse=True)
dataframes = []
for i in range(len(models)):
    feat_importances = pandas.DataFrame(sorted(zip(models[i].feature_importances_, feature_cols)),  columns=['Value', 'Feature'])
    dataframes.append(feat_importances)

feat_importances_sum = reduce(lambda x, y: x.add(y, fill_value=0), dataframes)
print(feat_importances_sum)
sns_feat_imp = plt.figure(figsize=(20, 10))
sns.barplot(x="Value", y='Feature', data=feat_importances_sum.sort_values(by="Value", ascending=False))
plt.title('LightGBM Features (avg over folds)')
plt.show()

feature_imp = pandas.DataFrame(sorted(zip(model.feature_importances_, feature_cols)), columns=['Value', 'Feature'])
sns_feat_imp = plt.figure(figsize=(20, 10))
sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value", ascending=False))
plt.title('LightGBM Features (avg over folds)')
plt.show()


shap_values /= 38
shap.summary_plot(shap_values, plot_type='bar', feature_names=feature_cols, show=True)
shap_global_charts(models, 0, X_test, feature_cols)

# lightgbm.plot_tree(model, figsize=(20, 7), tree_index=0, dpi=300, show_info=feature_cols)
# df_trees = model._Booster.dump_model()["tree_info"]
# print(df_trees)

dice_data = dice_ml.Data(dataframe=X_train_df, continuous_features=feature_cols, outcome_name='IPFVSALTRO')

dice_model = dice_ml.Model(model=model, backend='sklearn')
exp = dice_ml.Dice(dice_data, dice_model, method='random')
e = exp.generate_counterfactuals(X_test, total_CFs=5, desired_class="opposite", permitted_range={'Neu%': [0, 13],
                                                                                                 'Lin%':[0, 10],
                                                                                                 'Genere':[0,1]})
e.cf_examples_list[35].visualize_as_dataframe(show_only_changes=True)
e.visualize_as_dataframe(show_only_changes=True)

shap.initjs()
sample_idx = 0

for sample_idx in range(len(X_train)):
    explainer = shap.Explainer(models[sample_idx], X_train, feature_names=feature_cols)
    shap_values = explainer.shap_values(X_train[sample_idx][:])
    print("Expected/Base Value: \n", explainer.expected_value)
    print("\n Shap Values for Sample %d: " % sample_idx, shap_values)
    print("\n")
    prediction = models[sample_idx].predict(X_train[sample_idx].reshape(1, -1))[0]
    print("Prediction From Model: ", prediction)
    print("Prediction From Adding SHAP Values to Base Value : ", explainer.expected_value + shap_values.sum())

    shap.decision_plot(explainer.expected_value, shap_values, X_train[[sample_idx]],
                       link='logit', feature_names=feature_cols, show=False)
    file_name = f'{sample_idx} decision plot.png'
    plt.subplots_adjust(left=0.232)
    plt.title(f'Sample: {X_id[sample_idx]} Predicted: {prediction}, Real value: {y_true[sample_idx]}', y=1.07)
    plt.savefig(file_name)
    plt.show()
    shap_values_for_wf = explainer(X)
    file_name = f'{sample_idx} wf plot.png'
    shap.waterfall_plot(shap_values_for_wf[sample_idx], max_display=len(feature_cols), show=False)
    plt.subplots_adjust(left=0.300)
    plt.title(f'Sample: {X_id[sample_idx]} \n Predicted: {prediction}, Real value: {y_true[sample_idx]}', x=-0.15,
              y=1.05)
    plt.savefig(file_name)
    plt.show()
    sample_idx += 1
