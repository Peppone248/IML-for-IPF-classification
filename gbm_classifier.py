import pandas
import matplotlib.pyplot as plt
import shap
from IPython.core.display_functions import display
from IPython.display import Image
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV, LeaveOneOut, cross_val_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score, classification_report
from grid_search_utils import plot_grid_search, table_grid_search

pandas.set_option('display.max_columns', None)
data = pandas.read_csv('new Data Set Fibrosi.csv')
df = pandas.DataFrame(data)

df['RAPIDAVSLENTA'].replace(['LENTA', 'RAPIDA'], [0, 1], inplace=True)
df['IPFVSALTRO'].replace(['ALTRO', 'IPF'], [0, 1], inplace=True)
df['Patologia'].replace(['ALTRO', 'IPF', 'HP', 'NSIP'], [0, 1, 2, 3], inplace=True)
df['Genere'].replace(['M', 'F'], [0, 1], inplace=True)
df['Fumo'].replace(['si', 'no'], [0, 1], inplace=True)

df = df.drop('ID Lab', axis=1)
df = df.drop('Patologia', axis=1)

df = df.dropna()

feature_cols = ['Fumo', 'FVC%', 'FEV1%', 'DLCO', 'DLCO/VA', 'Macro%', 'Neu%', 'Lin%',
                'Età', 'RAPIDAVSLENTA', '2-DDCT KL-6', 'Genere',
                '2-DDCT MIR21', 'FEV1%/FVC%', '2-DDCT MIR92A']

# X_not_converted = df[feature_cols]
X = df[feature_cols].values
y = df['IPFVSALTRO'].values

cv = LeaveOneOut()
cv.get_n_splits(X)
y_true, y_pred = list(), list()
X_train, X_test = [], []
y_train, y_test = [], []

for train_ix, test_ix in cv.split(X):
    # split data
    X_train, X_test = X[train_ix, :], X[test_ix, :]
    y_train, y_test = y[train_ix], y[test_ix]
    model = GradientBoostingClassifier(random_state=42)
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

cm = confusion_matrix(y_true, y_pred, labels=model.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['ALTRO', 'IPF'])
matrix = disp.plot()
matrix.plot()
plt.show()