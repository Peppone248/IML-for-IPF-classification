import numpy as np
import pandas
import seaborn as sns
import matplotlib.pyplot as plt
import scikitplot as skplot
import shap
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, LeaveOneOut
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score, classification_report
from sklearn.preprocessing import MinMaxScaler
from useful_methods import features_encoding, shap_graphs_logreg, plot_conf_matrix, GSCV_tuning_model, shap_global_charts
import dice_ml
from functools import reduce
import scipy.spatial as sp
import scipy

plt.rcParams["axes.grid"] = False
pandas.set_option('display.max_columns', None)
data = pandas.read_csv('new Data Set Fibrosi.csv')
df = pandas.DataFrame(data)

features_encoding(df)

df = df.dropna()

feature_cols = ['Età', 'Genere', 'Fumo', 'FEV1%', 'DLCO/VA', 'Neu%', 'Lin%',
                '2-DDCT KL-6', '2-DDCT MIR21', '2-DDCT MIR92A', 'FEV1%/FVC%']

classes = ['ALTRO', 'IPF']

X_display = df[feature_cols]
X = df[feature_cols].values
y = df['IPFVSALTRO'].values
X_id = df['ID Lab'].values
X_not_scaled = X

"""scaler = MinMaxScaler(feature_range=(0, 1))
scaler.fit(X)"""
X = df[feature_cols].values

cv = LeaveOneOut()

y_true, y_pred = list(), list()
X_train, X_test = [], []
y_train, y_test = [], []

for train_ix, test_ix in cv.split(X):
    # split data
    X_train, X_test = X[train_ix, :], X[test_ix, :]
    y_train, y_test = y[train_ix], y[test_ix]
    model = LogisticRegression(random_state=42)
    model.fit(X_train, y_train)
    # evaluate model
    ytemp = model.predict(X_test)
    # store
    y_true.append(y_test[0])
    y_pred.append(ytemp[0])
    print(classification_report(y_true, y_pred, zero_division=1))
    print(len(y_true))
    accuracy = accuracy_score(y_true, y_pred)
    print("Accuracy: %.2f%%" % (accuracy * 100.0))
    print("real value ", y_true)
    print("predicted ", y_pred)

print(model.coef_)
print(np.std(X, 0)*model.coef_)

plot_conf_matrix(model, y_true, y_pred, classes)
plt.grid(None)
plt.show()

GSCV_tuning_model(X,y,cv, parameters = {
                'solver': ['liblinear'],
                'penalty': ['l1', 'l2'],
                'C': [0.98, 1.00, 1.02],
                'intercept_scaling': [1, 2, 3, 5, 8, 13, 21]
            })

y_true, y_pred = list(), list()
X_train, X_test = [], []
y_train, y_test = [], []
list_test_sets = list()
shap_values = None
models = []
list_shap_values = list()

for train_ix, test_ix in cv.split(X):
    # split data
    X_train, X_test = X[train_ix, :], X[test_ix, :]
    y_train, y_test = y[train_ix], y[test_ix]
    model = LogisticRegression(random_state=42,
                               penalty='l2',
                               solver='liblinear',
                               max_iter=200,
                               tol=1e-05,
                               intercept_scaling=2,
                               C=0.98)
    model.fit(X_train, y_train)
    # evaluate model
    ytemp = model.predict(X_test)
    # store
    y_true.append(y_test[0])
    y_pred.append(ytemp[0])
    models.append(model)
    print(classification_report(y_true, y_pred, zero_division=1))
    accuracy = accuracy_score(y_true, y_pred)
    print("Accuracy: %.2f%%" % (accuracy * 100.0))
    print("real value ", y_true)
    print("predicted ", y_pred)
    print("X_test: ", len(X_test))
    print("X_train: ", len(X_train))
    explainer = shap.Explainer(model, X_train)
    if shap_values is None:
        shap_values = explainer.shap_values(X_test)
    else:
        shap_values += explainer.shap_values(X_test)
    list_shap_values.append(shap_values)
    list_test_sets.append(test_ix)

# X_train_not_sc = scaler.inverse_transform(X_train)
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
shap_values /= 38
print('shap v after rounding: ', shap_values)


plot_conf_matrix(model, y_true, y_pred, classes)
plt.show()

print(model.coef_)
print(np.std(X, 0)*model.coef_)
coefficients = model.coef_[0]
dataframes = []
for i in range(len(models)):
    feat_importances = pandas.DataFrame(models[i].coef_[0], index=X_display.columns, columns=["Importance"])
    dataframes.append(feat_importances)

feat_importances_sum = reduce(lambda x,y: x.add(y, fill_value=0), dataframes)
print(feat_importances_sum)


feat_importances_sum.sort_values(by='Importance', ascending=False, inplace=True)
feat_importances_sum.plot(kind='bar', figsize=(9, 7))
plt.show()
plt.show()

dice_data = dice_ml.Data(dataframe=X_train_df, continuous_features=feature_cols, outcome_name='IPFVSALTRO')
dice_model = dice_ml.Model(model=model, backend='sklearn')
exp = dice_ml.Dice(dice_data, dice_model, method='random')
e = exp.generate_counterfactuals(X_test, total_CFs=5, desired_class="opposite")
e.cf_examples_list[2].visualize_as_dataframe(show_only_changes=True)
e.cf_examples_list[3].visualize_as_dataframe(show_only_changes=True)
e.visualize_as_dataframe(show_only_changes=True)

shap.initjs()
shap.summary_plot(shap_values, plot_type='bar', feature_names=feature_cols, show=True)
shap_global_charts(models, 0, X_test, feature_cols)
sample_idx = 0
# shap_global_interpret_graphs(model, X, X_train, feature_cols)
for sample_idx in range(len(X_train)):
    shap_graphs_logreg(models[sample_idx], X_not_scaled, X, X_train, feature_cols, sample_idx, y_true, X_id)
    sample_idx += 1
