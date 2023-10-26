import matplotlib.pyplot as plt
import numpy as np
import shap
from shap import maskers
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from grid_search_utils import plot_grid_search, table_grid_search


def features_encoding(df):
    df['RAPIDAVSLENTA'].replace(['LENTA', 'RAPIDA'], [0, 1], inplace=True)
    df['IPFVSALTRO'].replace(['ALTRO', 'IPF'], [0, 1], inplace=True)
    df['Patologia'].replace(['ALTRO', 'IPF', 'HP', 'NSIP'], [0, 1, 2, 3], inplace=True)
    df['Genere'].replace(['M', 'F'], [0, 1], inplace=True)
    df['Fumo'].replace(['si', 'no'], [0, 1], inplace=True)


def shap_graphs_logreg(model, X_not_scaled, X, X_train, feature_cols, sample_idx, y_true):
    explainer = shap.Explainer(model, X_train, feature_names=feature_cols)
    shap_values = explainer.shap_values(X_train[sample_idx][:])
    print("Expected/Base Value: \n", explainer.expected_value)
    print("\n Shap Values for Sample %d: " % sample_idx, shap_values)
    print("\n")
    prediction = model.predict(X_train[sample_idx].reshape(1, -1))[0]
    print("Prediction From Model: ", prediction)
    print("Prediction From Adding SHAP Values to Base Value : ", explainer.expected_value + shap_values.sum())

    shap.decision_plot(explainer.expected_value, shap_values, X_not_scaled[[sample_idx]],
                       link='logit', feature_names=feature_cols, show=False)
    file_name = f'{sample_idx} decision plot'
    plt.subplots_adjust(left=0.252)
    plt.title(f'Sample: {sample_idx} \n Predicted: {prediction}, Real value: {y_true[sample_idx]}', y=1.05)
    plt.savefig(file_name)
    plt.show()
    shap_values_for_wf = explainer(X)
    shap.waterfall_plot(shap_values_for_wf[sample_idx], max_display=len(feature_cols), show=False)
    file_name = f'{sample_idx} wf plot'
    plt.subplots_adjust(left=0.300)
    plt.title(f'Sample: {sample_idx} \n Predicted: {prediction}, Real value: {y_true[sample_idx]}', y=1.05)
    plt.savefig(file_name)
    plt.show()


def shap_graphs_lgbm(model, X, X_train, feature_cols, sample_idx, y_true, y_pred):
    explainer = shap.Explainer(model, X_train, feature_names=feature_cols)
    shap_values = explainer.shap_values(X_train[sample_idx][:])
    print("Expected/Base Value: \n", explainer.expected_value)
    print("\n Shap Values for Sample %d: " % sample_idx, shap_values)
    print("\n")
    print("Prediction From Model: ", y_pred[sample_idx])
    print("Prediction From Adding SHAP Values to Base Value : ", explainer.expected_value + shap_values.sum())
    shap.bar_plot(explainer.shap_values(
        X_train[sample_idx][:]),
        feature_names=feature_cols,
        max_display=len(feature_cols), show=False)
    file_name = f'bar plot{sample_idx}.png'
    plt.subplots_adjust(left=0.252)
    plt.title(f'Predicted: {y_pred[sample_idx]}, Real value: {y_true[sample_idx]}', y=1.05)
    plt.savefig(file_name)
    plt.show()
    shap_values_for_wf = explainer(X_train)
    file_name = f'wf plot{sample_idx}.png'
    shap.waterfall_plot(shap_values_for_wf[sample_idx], max_display=len(feature_cols), show=False)
    plt.subplots_adjust(left=0.300)
    plt.title(f'Predicted: {y_pred[sample_idx]}, Real value: {y_true[sample_idx]}', y=1.05)
    plt.savefig(file_name)
    plt.show()


def shap_global_charts(models, sample_idx, X_test, feature_cols):
    shap_exp = None
    for i in range(0, len(models)):
        explainer = shap.Explainer(models[sample_idx], X_test, feature_names=feature_cols)
        if shap_exp is None:
            shap_exp = explainer(X_test)
        else:
            shap_exp += explainer(X_test)

    shap_exp /= 38
    shap.plots.heatmap(shap_exp, max_display=len(feature_cols))
    shap.plots.bar(shap_exp, show=False)
    plt.xlabel('mean(|SHAP value|)')
    plt.show()
    sample_idx += 1


def shap_global_charts_tree_exp(models, sample_idx, X_test, feature_cols):
    shap_exp = None
    for i in range(0, len(models)):
        explainer = shap.Explainer(models[sample_idx], X_test, feature_names=feature_cols)
        if shap_exp is None:
            shap_exp = explainer(X_test)
        else:
            shap_exp += explainer(X_test)

    shap_exp /= 38
    shap.plots.heatmap(shap_exp[:,:,0], max_display=len(feature_cols))
    shap.plots.bar(shap_exp[:,:,0], show=False)
    plt.xlabel('mean(|SHAP value|)')
    plt.show()
    sample_idx += 1


def shap_graphs_decision_tree(model, X, X_train, feature_cols, sample_idx, y_true, X_id):
    explainer = shap.TreeExplainer(model, X_train)
    shap_values = explainer.shap_values(X_train)
    # Expected/Base/Reference value = the value that would be predicted if we didn’t know any features of the current output”
    print('Expected Value:', explainer.expected_value)
    prediction = model.predict(X_train[sample_idx].reshape(1, -1))[0]
    print("Prediction From Model: ", prediction)
    # print("Prediction From Adding SHAP Values to Base Value : ", explainer.expected_value + sum(shap_values))

    shap.initjs()
    # shap.summary_plot(shap_values[1], X_train, plot_type='bar', feature_names=feature_cols)
    # shap.decision_plot(explainer.expected_value[1], shap_values[1], feature_names=feature_cols)
    explainer = shap.Explainer(model, X_train, feature_names=feature_cols)
    shap_values = explainer(X, check_additivity=False)
    shap.plots.waterfall(shap_values[:, :, 1][sample_idx, :], max_display=len(feature_cols), show=False)
    file_name = f'{sample_idx} wf plot'
    plt.subplots_adjust(left=0.300)
    plt.title(f'Sample: {X_id[sample_idx]} \n Predicted: {prediction}, Real value: {y_true[sample_idx]}', y=1.05)
    plt.savefig(file_name)
    plt.show()
    shap.decision_plot(explainer.expected_value[1], explainer.shap_values(X)[1][sample_idx], X[[sample_idx]],
                       link='logit', feature_names=feature_cols, show=False)
    file_name = f'{sample_idx} decision plot'
    plt.subplots_adjust(left=0.252)
    plt.title(f'Sample: {X_id[sample_idx]} \n Predicted: {prediction}, Real value: {y_true[sample_idx]}', y=1.05)
    plt.savefig(file_name)
    plt.show()
    """ 
    shap.decision_plot(explainer.expected_value[1], shap_values[1][sample_idx], X_train[[sample_idx]],
                       link='logit', feature_names=feature_cols, show=False)
    file_name = f'{sample_idx} decision plot.png'
    plt.subplots_adjust(left=0.252)
    plt.title(f'Predicted: {prediction}, Real value: {y_true[sample_idx]}', y=1.05)
    plt.savefig(file_name)
    plt.show()
    shap.plots._waterfall.waterfall_legacy(explainer.expected_value[1], shap_values[1][sample_idx],
                                           feature_names=feature_cols, max_display=len(feature_cols), show=False)
    file_name = f'{sample_idx} wf plot.png'
    plt.subplots_adjust(left=0.220)
    plt.title(f'Predicted: {prediction}, Real value: {y_true[sample_idx]}', y=1.05)
    plt.savefig(file_name)
    plt.show()"""
    # shap.plots.heatmap(shap_values, max_display=len(feature_cols))


def plot_conf_matrix(model, y_true, y_pred, classes):
    cm = confusion_matrix(y_true, y_pred, labels=model.classes_)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
    matrix = disp.plot()


def GSCV_tuning_model(X, y, cv, parameters):
    choice = 0
    y_true, y_pred = list(), list()
    X_train, X_test = [], []
    y_train, y_test = [], []
    print("You want to perform tuning for this model? \n Y/N")
    var = input().upper()
    if var == 'Y':
        print(
            "For which model do you want to perform the tuning?\n 1 for DecisionTree \n 2 for LogReg \n 3 for XGB \n 4 for LGBM")
        choice = int(input())
    if var == 'Y' and choice == 1:
        for train_ix, test_ix in cv.split(X):
            parameters
            # split data
            X_train, X_test = X[train_ix, :], X[test_ix, :]
            y_train, y_test = y[train_ix], y[test_ix]
            tune_model = DecisionTreeClassifier(random_state=42)
            grid = GridSearchCV(tune_model, parameters, cv=cv, verbose=1, n_jobs=-1)
            grid.fit(X_train, y_train)

        plot_grid_search(grid)
        table_grid_search(grid, all_ranks=True)
    elif var == 'Y' and choice == 2:
        for train_ix, test_ix in cv.split(X):
            parameters
            # split data
            X_train, X_test = X[train_ix, :], X[test_ix, :]
            y_train, y_test = y[train_ix], y[test_ix]
            tune_model = LogisticRegression(random_state=42)
            grid = GridSearchCV(tune_model, parameters, cv=cv, verbose=1, n_jobs=-1)
            grid.fit(X_train, y_train)

        plot_grid_search(grid)
        table_grid_search(grid, all_ranks=True)

    elif var == 'Y' and choice == 3:
        for train_ix, test_ix in cv.split(X):
            parameters
            # split data
            X_train, X_test = X[train_ix, :], X[test_ix, :]
            y_train, y_test = y[train_ix], y[test_ix]
            tune_model = XGBClassifier(random_state=42)
            grid = GridSearchCV(tune_model, parameters, cv=cv, verbose=1, n_jobs=-1)
            grid.fit(X_train, y_train)

        plot_grid_search(grid)
        table_grid_search(grid, all_ranks=True)

    elif var == 'Y' and choice == 4:
        for train_ix, test_ix in cv.split(X):
            parameters
            # split data
            X_train, X_test = X[train_ix, :], X[test_ix, :]
            y_train, y_test = y[train_ix], y[test_ix]
            tune_model = LGBMClassifier(random_state=42, verbose=-1, min_data_leaf=2)
            grid = GridSearchCV(tune_model, parameters, cv=cv, verbose=1, n_jobs=-1)
            grid.fit(X_train, y_train)

        plot_grid_search(grid)
        table_grid_search(grid, all_ranks=True)
    else:
        print("check")
