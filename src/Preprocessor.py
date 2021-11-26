# author: Abhiket Gaurav
# date: 2021-11-25

"""Reads train csv data from path, preprocess the data, build a Model, gives the cross validation output"
Usage: Preprocessor.py --path=<path> --score_file=<score_file> --model_path=<model_path>
 
Options:
--path=<path>               Path to read file from
--score_file=<score_file>   Path (including filename) of where to locally save cross val score
--model_path=<model_path>   Path for the model pickle file
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


opt = docopt(__doc__)


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

    preprocessor = make_column_transformer(
        (make_pipeline(SimpleImputer(), StandardScaler()),numeric_features,),
        (OrdinalEncoder(),ordinal_features,),  
        ("passthrough", passthrough_features),)

    # Building the Model
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
        
    # Output: saving the cross val score in a CSV file
    
    try:
        pd.DataFrame(results_bal_f).to_csv(out_file)
    except:
        os.makedirs(os.path.dirname(out_file))
        pd.DataFrame(results_bal_f).to_csv(out_file)

if __name__ == "__main__":
    main(opt["--path"], opt["--score_file"],opt["--model_path"])



