# author : Valli Akella
# date: 2021-11-25

"""This code predicts the contraceptive method used based on the model derived from a pickle file and generates PR curve, Roc
curve, classification report and confusion matrix.
Usage: predict.py --test_path=<test_path>, --model=<model>, --output_path=<output_path>

Options:
--test_path=<test_path>        path to the X test
--model=<model>                pickle file containing to predict the targets of the testdata
--output_path=<output_path>    path to save the files from the script locally
 """


import sys
import time
import os
import string

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    
    classification_report,
    confusion_matrix,
    roc_curve
)
from sklearn.metrics import PrecisionRecallDisplay 
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.impute import SimpleImputer
from docopt import docopt

#%matplotlib inline
import pickle

opt = docopt(__doc__)

def plot_ROC_curve(y_true, y_pred):
    label="SVC: ROC Curve",
    marker_colour="r",
    marker_label="SVC default threshold",
    fpr, tpr, thresholds = roc_curve(y_true, y_pred)
    plt.plot(fpr, tpr, label=label)
    plt.xlabel("FPR")
    plt.ylabel("TPR (recall)")

    default_threshold = np.argmin(np.abs(thresholds - 0.5))

    plt.plot(
        fpr[default_threshold],
        tpr[default_threshold],
        "or",
        markersize=10,
        label="threshold 0.5",
    )
    plt.legend(loc="best");

def main(test_path,model, output_path):
    test_df = pd.read_csv(test_path)

    #splitting into X_test and y_test
    X_test, y_test = test_df.drop(columns=["Contraceptive_method_used"]), test_df["Contraceptive_method_used"]

    #Converting it to a binary model(test set)
    y_test = y_test.replace(1,0)
    y_test = y_test.replace([2,3],1)

    final_svc_model = pickle.load(open("/Users/valliakella/Documents/UBC/522Project/contraceptive_method_predictor/results/models/final_svc.pkl", "rb"))

    #Predictions on the test data
    y_pred = final_svc_model.predict(X_test)
    pred_df = pd.DataFrame(y_pred, y_test)


    #Confusion Matrix
    confusion_matrix_report = confusion_matrix(y_test, y_pred)

    #Classification Report
    classificationreport = classification_report(
        y_test, y_pred, target_names=["contra_no", "contra_yes"], output_dict=True)
    

    try:
        pd.DataFrame(confusion_matrix_report).to_csv(output_path+"confusion_matrix.csv")
    except:
        os.makedirs(os.path.dirname(output_path))
        pd.DataFrame(confusion_matrix_report).to_csv(output_path+"confusion_matrix.csv")
    
    try:
        pd.DataFrame(classificationreport).to_csv(output_path+"classification_report.csv")
    except:
        os.makedirs(os.path.dirname(output_path))
        pd.DataFrame(classificationreport).to_csv(output_path+"classification_report.csv")
    

    # Generate PR Curve
    pr_curve = PrecisionRecallDisplay.from_estimator(final_svc_model, X_test, y_test)
    plt.savefig(output_path +"pr_curve.png")

    # Generate ROC Curve
    
    roc_plot = plot_ROC_curve(y_test, final_svc_model.predict_proba(X_test)[:, 1])
    plt.savefig(output_path +"roc_curve.png")



if __name__ == "__main__":
    main(opt["--test_path"], opt["--model"],opt["--output_path"])


     
