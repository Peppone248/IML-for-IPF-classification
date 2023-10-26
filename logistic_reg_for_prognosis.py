import numpy as np
import pandas
import seaborn as sns
import matplotlib.pyplot as plt
import shap
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, LeaveOneOut
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score, classification_report
from sklearn.preprocessing import MinMaxScaler
from grid_search_utils import plot_grid_search, table_grid_search
from useful_methods import features_encoding, GSCV_tuning_model, shap_graphs_logreg, plot_conf_matrix

pandas.set_option('display.max_columns', None)
data = pandas.read_csv('new Data Set Fibrosi.csv')
df = pandas.DataFrame(data)

features_encoding(df)

df = df.dropna()

feature_cols = ['2-DDCT KL-6', '2-DDCT MIR21', '2-DDCT MIR92A']

classes = ['LENTA', 'RAPIDA']

X = df[feature_cols].values
y = df['RAPIDAVSLENTA'].values

scaler = MinMaxScaler(feature_range=(0, 1))
scaler.fit(X)
X = scaler.transform(df[feature_cols].values)

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
    accuracy = accuracy_score(y_true, y_pred)
    print("Accuracy: %.2f%%" % (accuracy * 100.0))
    print("real value ", y_true)
    print("predicted ", y_pred)

print(model.coef_)
print(np.std(X, 0) * model.coef_)

cm = confusion_matrix(y_true, y_pred, labels=model.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['LENTA', 'RAPIDA'])
matrix = disp.plot()
plt.show()

y_true, y_pred = list(), list()
X_train, X_test = [], []
y_train, y_test = [], []

GSCV_tuning_model(X, y, cv, parameters={'solver': ['liblinear'],
                                        #'intercept_scaling': [1, 2, 3, 5, 8, 13, 21, 34],
                                        'C': [(0.9 + x / 50) for x in range(0, 10)],
                                        'tol': [1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10, 1e-11],
                                        })

y_true, y_pred = list(), list()
X_train, X_test = [], []
y_train, y_test = [], []
for train_ix, test_ix in cv.split(X):
    # split data
    X_train, X_test = X[train_ix, :], X[test_ix, :]
    y_train, y_test = y[train_ix], y[test_ix]
    model = LogisticRegression(random_state=42,
                               penalty='l2',
                               solver='liblinear',
                               max_iter=300,
                               C=0.98,
                               tol=1e-05,
                               intercept_scaling=1,
                               class_weight='balanced')
    model.fit(X_train, y_train)
    # evaluate model
    ytemp = model.predict(X_test)
    # store
    y_true.append(y_test[0])
    y_pred.append(ytemp[0])
    print(classification_report(y_true, y_pred, zero_division=1))
    accuracy = accuracy_score(y_true, y_pred)
    print("Accuracy: %.2f%%" % (accuracy * 100.0))
    print("real value ", y_true)
    print("predicted ", y_pred)

plot_conf_matrix(model, y_true, y_pred, classes)

print(model.coef_)
print(np.std(X, 0) * model.coef_)
coefficients = model.coef_[0]

feature_importance = pandas.DataFrame({'Feature': df[feature_cols].columns, 'Importance': np.abs(coefficients)})
feature_importance = feature_importance.sort_values('Importance', ascending=True)
feature_importance.plot(x='Feature', y='Importance', kind='barh', figsize=(10, 6))
plt.show()

"""shap.initjs()
explainer = shap.LinearExplainer(model, X_train, feature_names=feature_cols)
shap_values = explainer(X_test)
shap.plots.bar(shap_values, max_display=14)
shap.force_plot(explainer.expected_value, shap_values[0].values, X_test, feature_names=feature_cols, matplotlib=True, show=True)
shap.summary_plot(shap_values, X_test, plot_type="bar", feature_names=feature_cols)
shap.plots.waterfall(shap_values[0], max_display=14)
"""
