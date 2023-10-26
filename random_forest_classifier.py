import pandas
import graphviz
import matplotlib.pyplot as plt
import shap
from IPython.core.display_functions import display
from IPython.display import Image
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, LeaveOneOut, cross_val_score
from sklearn.tree import export_graphviz
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score, classification_report
from grid_search_utils import plot_grid_search, table_grid_search
from sklearn.tree import export_graphviz

pandas.set_option('display.max_columns', None)
data = pandas.read_csv('new Data Set Fibrosi.csv')
df = pandas.DataFrame(data)

df['RAPIDAVSLENTA'].replace(['LENTA', 'RAPIDA'], [0, 1], inplace=True)
df['IPFVSALTRO'].replace(['ALTRO', 'IPF'], [0, 1], inplace=True)
df['Patologia'].replace(['ALTRO', 'IPF', 'HP', 'NSIP'], [0, 1, 2, 3], inplace=True)
df['Genere'].replace(['M', 'F'], [0, 1], inplace=True)
df['Fumo'].replace(['si', 'no'], [0, 1], inplace=True)

df = df.drop('ID Lab', axis=1)

df = df.dropna()

feature_cols = ['Fumo', 'FVC%', 'FEV1%', 'DLCO', 'DLCO/VA', 'Macro%', 'Neu%', 'Lin%',
                'Età', '2-DDCT KL-6', '2-DDCT MIR21', 'FEV1%/FVC%', '2-DDCT MIR92A', 'IPFVSALTRO']

classes=['LENTA', 'RAPIDA']

X_not_converted = df[feature_cols]
X = df[feature_cols].values
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
    model = RandomForestClassifier(random_state=42)
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

cm = confusion_matrix(y_true, y_pred, labels=model.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
matrix = disp.plot()

tree = model.estimators_[0]
dot_data = export_graphviz(tree,
                           feature_names=feature_cols,
                           filled=True,
                           max_depth=2,
                           impurity=False,
                           proportion=True)
graph = graphviz.Source(dot_data)
graph.render('dtree_render_random_forest', view=True)

tree = model.estimators_[1]
dot_data = export_graphviz(tree,
                           feature_names=feature_cols,
                           filled=True,
                           max_depth=2,
                           impurity=False,
                           proportion=True)
graph = graphviz.Source(dot_data)
graph.render('dtree_render_random_forest1', view=True)

tree = model.estimators_[2]
dot_data = export_graphviz(tree,
                           feature_names=feature_cols,
                           filled=True,
                           max_depth=2,
                           impurity=False,
                           proportion=True)
graph = graphviz.Source(dot_data)
graph.render('dtree_render_random_forest2', view=True)

importances = model.feature_importances_
feat_labels = feature_cols

feat_importances = pandas.DataFrame(model.feature_importances_, index=X_not_converted.columns, columns=["Importance"])
feat_importances.sort_values(by='Importance', ascending=False, inplace=True)
feat_importances.plot(kind='bar', figsize=(9, 7))
plt.show()

y_true, y_pred = list(), list()
X_train, X_test = [], []
y_train, y_test = [], []

for train_ix, test_ix in cv.split(X):
    parameters = {
        'n_estimators': [200, 250, 350],
        'max_depth': [2, 4, 5, 8, 10],
        # 'max_features': [2, 4, 8, 10],
        'criterion': ['gini', 'entropy']
    }
    # split data
    X_train, X_test = X[train_ix, :], X[test_ix, :]
    y_train, y_test = y[train_ix], y[test_ix]
    tune_model = RandomForestClassifier(random_state=42)
    grid = GridSearchCV(tune_model, parameters, cv=cv, verbose=1, n_jobs=-1)
    grid.fit(X_train, y_train)

plot_grid_search(grid)
table_grid_search(grid, all_ranks=True)

y_true, y_pred = list(), list()
X_train, X_test = [], []
y_train, y_test = [], []
for train_ix, test_ix in cv.split(X):
    # split data
    X_train, X_test = X[train_ix, :], X[test_ix, :]
    y_train, y_test = y[train_ix], y[test_ix]
    model = RandomForestClassifier(criterion='gini',
                                   min_samples_split=3,
                                   random_state=42,
                                   max_depth=5,
                                   max_features=5,
                                   n_estimators=250,
                                   bootstrap=True,
                                   min_samples_leaf=8
                                   )
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

tree = model.estimators_[0]
dot_data = export_graphviz(tree,
                           feature_names=feature_cols,
                           filled=True,
                           max_depth=model.max_depth,
                           impurity=False,
                           proportion=True)
graph = graphviz.Source(dot_data)
graph.render('dtree_render_random_forest_tuning', view=True)

tree = model.estimators_[1]
dot_data = export_graphviz(tree,
                           feature_names=feature_cols,
                           filled=True,
                           max_depth=model.max_depth,
                           impurity=False,
                           proportion=True)
graph = graphviz.Source(dot_data)
graph.render('dtree_render_random_forest_tuning1', view=True)

tree = model.estimators_[2]
dot_data = export_graphviz(tree,
                           feature_names=feature_cols,
                           filled=True,
                           max_depth=model.max_depth,
                           impurity=False,
                           proportion=True)
graph = graphviz.Source(dot_data)
graph.render('dtree_render_random_forest_tuning2', view=True)

tree = model.estimators_[3]
dot_data = export_graphviz(tree,
                           feature_names=feature_cols,
                           filled=True,
                           max_depth=model.max_depth,
                           impurity=False,
                           proportion=True)
graph = graphviz.Source(dot_data)
graph.render('dtree_render_random_forest_tuning3', view=True)

importances = model.feature_importances_
feat_labels = feature_cols

feat_importances = pandas.DataFrame(model.feature_importances_, index=X_not_converted.columns, columns=["Importance"])
feat_importances.sort_values(by='Importance', ascending=False, inplace=True)
feat_importances.plot(kind='bar', figsize=(9, 7))
plt.show()

