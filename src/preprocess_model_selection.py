# author: Abhiket Gaurav, Christopher Alexander
# date: 2021-11-25

"""Reads train csv data from path, preprocess the data, build a Model, gives the cross validation output"
Usage: preprocess_model_selection.py --path=<path> --score_file=<score_file> [--model_path=<model_path>]
 
Options:
--path=<path>               Path to read file from
--score_file=<score_file>   Path (including filename) of where to locally save cross val score
--model_path=<model_path>   Path for the model pickle file [default: ../results/models/final_svc.pkl]
"""

import os
import pandas as pd
from docopt import docopt
import numpy as np
import sys
from sklearn.model_selection import train_test_split
from sklearn.dummy import DummyClassifier, DummyRegressor

from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import (
    OneHotEncoder,
    OrdinalEncoder,
    PolynomialFeatures,
    StandardScaler,
)

from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier
from sklearn.compose import ColumnTransformer, make_column_transformer
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.feature_extraction.text import CountVectorizer

from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import (
    GridSearchCV,
    RandomizedSearchCV,
    ShuffleSplit,
    cross_val_score,
    cross_validate,
    train_test_split,
)
from sklearn.svm import SVC, SVR
import pickle


opt = docopt(__doc__)

def make_preprocessor(numeric_features, ordinal_features, passthrough_features):
    """
    Creates and returns a preprocessor column transformer.
    Makes a column transformer of standard scaler, OHE and passthrough.

    Parameters
    ----------
    numeric_features : list
        Column names of numeric features 
    ordinal_features : list
        Column names of ordinal features
    passthrough_features : list
        Column names of features which does not require preprocessing

    Returns
    -------
    preprocessor
        A column transformer object with specific transformations for each set of columns.

    Examples
    --------
    >>> make_preprocessor(['Wife_age'], ["Wife_education","Husband_education"], ["Wife_religion"])
    """
    preprocessor = make_column_transformer(
        (make_pipeline(SimpleImputer(), StandardScaler()),numeric_features,),
        (OrdinalEncoder(),ordinal_features,),  
        ("passthrough", passthrough_features),)
    return preprocessor

def cross_val_multiple_models(preprocessor, X_train, y_train):
    """
    Performs cross validate on multiple models and creats a result dataframe

    Parameters
    ----------
    preprocessor : make_column transformer object
        The column transformer object to be added to pipeline
    X_train : dataframe
        Dataset to train on 
    y_train : Pandas series
        Target values to train on

    Returns
    -------
    results_bal_f
        Returns a dataframe with cross validation scores

    Examples
    --------
    >>> cross_val_multiple_models(preprocessor, X_train, y_train)
    """
    models_bal = {
        "decision tree": DecisionTreeClassifier(random_state=123),
        "kNN": KNeighborsClassifier(),
        "Logistic Regression": LogisticRegression(max_iter=100, multi_class='ovr',random_state=123 ),
        "RBF SVM": SVC(random_state=123),}
    #Cross val-score
    results_bal = {}
    results_bal_f = {}
    for keys, value in models_bal.items():
        pipe_bal = make_pipeline(preprocessor, value)
        results_bal[keys] = cross_validate(pipe_bal, X_train, y_train, cv = 5, return_train_score = True)
        results_bal_f[keys] = pd.DataFrame(results_bal[keys]).mean()
    return results_bal_f
    
def hyperparameter_tuning(preprocessor, X_train, y_train):
    """
    Performs hyperparameter tuning and returns best model and parameters

    Parameters
    ----------
    preprocessor : make_column transformer object
        The column transformer object to be added to pipeline 
    X_train : dataframe
        Dataset to train on 
    y_train : Pandas series
        Target values to train on

    Returns
    -------
    random_search.best_estimator_
        Returns the best estimator after RandomSearchCV. 
    
    random_search.best_params_
        Returns the best params in a dictionary.

    Examples
    --------
    >>> hyperparameter_tuning(preprocessor, X_train, y_train)
    """
    param = {
    "svc__class_weight": [None,"balanced"],
    "svc__gamma": np.logspace(-3, 0, 4),
    "svc__C": np.logspace(-2, 3, 6)
    } 
    pipe = make_pipeline(preprocessor, SVC(probability= True, random_state=123))
    random_search = RandomizedSearchCV(pipe, param, n_iter=200, verbose=1, n_jobs=-1, random_state=123)
    random_search.fit(X_train, y_train)
    cv_results_df = pd.DataFrame(random_search.cv_results_)[
    [
        "rank_test_score",
        "mean_test_score",
        "param_svc__gamma",
        "param_svc__C",
        "param_svc__class_weight",
        "mean_fit_time",
    ]].set_index("rank_test_score").sort_index().T
    print("Best hyperparameter values: ", random_search.best_params_)
    print("Best score: %0.3f" % (random_search.best_score_))
    return random_search.best_estimator_,random_search.best_params_
    



def main(path, out_file, model_path):
    # Reading the data
    train_df = pd.read_csv(path)
    
    # Splitting between Features & Tar
    X_train, y_train = train_df.drop(columns=["Contraceptive_method_used"]), train_df["Contraceptive_method_used"]

    #Converting it to a binary model(train set)
    y_train = y_train.replace(1,0)
    y_train = y_train.replace([2,3],1)

    #Cleaning and Pre-processing
    numeric_features = ['Wife_age', 'Number_of_children_ever_born']
    ordinal_features = ["Wife_education","Husband_education","Husband_occupation", "Standard_of_living_index"]
    passthrough_features = ['Wife_religion','Wife_now_working?','Media_exposure'] 

    preprocessor = make_preprocessor(numeric_features, ordinal_features, passthrough_features)
        
    results_bal_f = cross_val_multiple_models(preprocessor, X_train, y_train)
    # Output: saving the cross val score in a CSV file
    
    try:
        pd.DataFrame(results_bal_f).to_csv(out_file)
    except:
        os.makedirs(os.path.dirname(out_file))
        pd.DataFrame(results_bal_f).to_csv(out_file)

    # Model Tuning 
    # SVC was decided to be the best model for this scenario
    best_model, best_params = hyperparameter_tuning(preprocessor, X_train, y_train)

    try:
        pickle.dump(best_model, open(str(model_path),"wb"))
        pickle.dump(best_params, open(str(os.path.dirname(out_file))+"/final_params.pkl","wb"))
    except:
        os.makedirs(os.path.dirname(model_path))
        directory = os.path.dirname(model_path)
        pickle.dump(best_model, open(model_path,"wb"))
        pickle.dump(best_params, open(str(directory)+"/final_params.pkl","wb"))

    

if __name__ == "__main__":
    main(opt["--path"], opt["--score_file"],opt["--model_path"])



