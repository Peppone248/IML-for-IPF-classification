import pandas
import seaborn as sns
import matplotlib.pyplot as plt
import shap
import scikitplot as skplot
from IPython.core.display_functions import display
from IPython.display import Image
import lightgbm
from sklearn.model_selection import GridSearchCV, LeaveOneOut, cross_val_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score, classification_report
from sklearn.preprocessing import MinMaxScaler
from useful_methods import features_encoding, GSCV_tuning_model
from grid_search_utils import plot_grid_search, table_grid_search

pandas.set_option('display.max_columns', None)
data = pandas.read_csv('new Data Set Fibrosi.csv')
df = pandas.DataFrame(data)

features_encoding(df)

df = df.drop('ID Lab', axis=1)

feature_cols = ['2-DDCT KL-6', '2-DDCT MIR21', '2-DDCT MIR92A', 'Età', 'DLCO', 'FEV1%']

# X_not_converted = df[feature_cols]
X_display = df[feature_cols]
X = df[feature_cols].values
y = df['RAPIDAVSLENTA'].values

scaler = MinMaxScaler(feature_range=(0, 1))
scaler.fit(X)
X = scaler.transform(df[feature_cols].values)

cv = LeaveOneOut()
cv.get_n_splits(X)
y_true, y_pred = list(), list()
X_train, X_test = [], []
y_train, y_test = [], []

for train_ix, test_ix in cv.split(X):
    # split data
    X_train, X_test = X[train_ix, :], X[test_ix, :]
    y_train, y_test = y[train_ix], y[test_ix]
    model = lightgbm.LGBMClassifier(random_state=42, verbose=-1, min_data_leaf=2)
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

GSCV_tuning_model(X, y, cv, parameters={'boosting_type': ['gbdt', 'goss'],
                                        #'num_leaves': [34, 50, 55, 60, 89],
                                        'max_depth': [2, 4, 5, 10, 13],
                                        'learning_rate': [0.05, 0.2, 0.5, 0.7],
                                        #'n_estimators': [1900, 2000, 2100],
                                        #'subsample_for_bin': [610, 987, 1597],
                                        #'min_child_samples': [2, 3, 5, 8],
                                        #'colsample_bytree': [0.85, 0.9, 0.95],
                                        #'reg_alpha': [0.2, 0.7],
                                        'reg_lambda': [0.2, 0.5]
                                        })

y_true, y_pred = list(), list()
X_train, X_test = [], []
y_train, y_test = [], []
for train_ix, test_ix in cv.split(X):
    # split data
    X_train, X_test = X[train_ix, :], X[test_ix, :]
    y_train, y_test = y[train_ix], y[test_ix]
    model = lightgbm.LGBMClassifier(random_state=42,
                                    boosting_type='gbdt',
                                    learning_rate=0.05,
                                    min_data_in_leaf=8,
                                    max_depth=2,
                                    reg_lambda=0.4,
                                    verbose=-1
                                    )
    model.fit(X_train, y_train)
    # evaluate model
    ytemp = model.predict(X_test)
    # store
    y_true.append(y_test[0])
    y_pred.append(ytemp[0])
    # print(classification_report(y_true, y_pred, zero_division=1))
    scores = cross_val_score(model, X, y, scoring='accuracy', cv=cv, n_jobs=-1)
    accuracy = accuracy_score(y_true, y_pred)
    print("Accuracy: %.2f%%" % (accuracy * 100.0))
    print('Best score: {}'.format(scores))
    print("real value ", y_true)
    print("predicted ", y_pred)

lightgbm.plot_importance(model, importance_type='gain', ylabel=feature_cols)
sorted(zip(model.feature_importances_, feature_cols), reverse=True)
feature_imp = pandas.DataFrame(sorted(zip(model.feature_importances_, feature_cols)), columns=['Value', 'Feature'])
plt.figure(figsize=(20, 10))
sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value", ascending=False))
plt.title('LightGBM Features (avg over folds)')
plt.tight_layout()
plt.show()

# lightgbm.plot_tree(model, figsize=(20, 7), tree_index=0, dpi=300, show_info=feature_cols)
df_trees = model._Booster.dump_model()["tree_info"]
# print(df_trees)

cm = confusion_matrix(y_true, y_pred, labels=model.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['ALTRO', 'IPF'])
matrix = disp.plot()

shap.initjs()
explainer = shap.Explainer(model, X_train, feature_names=feature_cols)
sample_idx = 15
shap_values = explainer.shap_values(X_train[sample_idx][:])
print("Expected/Base Value: \n", explainer.expected_value)
print("\n Shap Values for Sample %d: " % sample_idx, shap_values)
print("\n")
print("Prediction From Model: ", model.predict(X_train[sample_idx][:].reshape(1, -1))[0])
print("Prediction From Adding SHAP Values to Base Value : ", explainer.expected_value + shap_values.sum())

shap.summary_plot(explainer.shap_values(X_train),
                  feature_names=feature_cols,
                  plot_type="bar",
                  )
shap.bar_plot(explainer.shap_values(
    X_train[sample_idx][:]),
    feature_names=feature_cols,
    max_display=len(feature_cols))
shap_values_for_wf = explainer(X_train)
shap.waterfall_plot(shap_values_for_wf[sample_idx], max_display=len(feature_cols))
shap.plots.bar(explainer(X_train), max_display=len(feature_cols))
shap.decision_plot(explainer.expected_value, explainer.shap_values(X_train), feature_names=feature_cols)
