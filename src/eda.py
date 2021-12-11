# author: Christopher Alexander, Harry Chan, Abhiket Gaurav,Valli A
# date: 2021-11-24

"""Create eda plots for the provided training data file. Saves the plots as png files in the provided output directory.
Usage: src/eda.py --train_dir=<train_dir> --out_dir=<out_dir>

Options:
--train_dir=<train_dir>    Path (including filename) to training data
--out_dir=<out_dir>    Path (including filename) of where to locally write the file
"""
  
from docopt import docopt
import altair as alt
import pandas as pd
import os 

opt = docopt(__doc__)

def main(train_dir, out_dir):

    # Read the training data
    train_df = pd.read_csv(train_dir)
    # Create the directory if not exists
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    # Distribution of target variable
    hist_target = alt.Chart(train_df, title="Histogram of the Contraceptive Method used").mark_bar().encode(
        x=alt.X("Contraceptive_method_used", title="Contraceptive method used", type="nominal"),
        y=alt.Y("count()", title="Count"),
    )
    hist_target.save(out_dir + "/histogram_target.png")

    # Specify Numerical and Categorical Features
    numerical_features = ["Wife_age", "Number_of_children_ever_born"]
    non_numerical_features = list(train_df.drop(columns=(numerical_features + ['Contraceptive_method_used'])).columns)

    # Plot the histograms for each numerical feature
    hist_num = alt.Chart(train_df).mark_bar().encode(
        x=alt.X(alt.repeat(), type="quantitative", bin=alt.Bin(maxbins=30)),
        y=alt.Y("count()", title="Count"),
    ).repeat(numerical_features, title="Histogram of all numerical features")
    hist_num.save(out_dir + "/histogram_numerical.png")

    # Plot the histograms for each categorical feature
    hist_cat = alt.Chart(train_df).mark_bar().encode(
        x=alt.X("count()", title="Count"),
        y=alt.Y(alt.repeat(), type="nominal"),
    ).repeat(non_numerical_features, columns=2, title="Histogram of all categorical features")
    hist_cat.save(out_dir + "/histogram_categorical.png")

    # Count combinations of categorical features
    counts_cat = alt.Chart(train_df).mark_bar().encode(
        x=alt.X("count()", title="Count"),
        y=alt.Y(alt.repeat(), type="nominal"),
    ).repeat(non_numerical_features, columns=2, title="Histogram of all categorical features")

    counts_cat.save(out_dir + "/counts_categorical_by_target.png")

    # Count combinations of categorical features by the target variable
    counts_cat_by_target = alt.Chart(train_df).mark_square().encode(
        x=alt.X(alt.repeat(), type="nominal"),
        y=alt.Y("Contraceptive_method_used:N", title="CMU"),
        color="count()",
        size="count()"
    ).repeat(
        non_numerical_features,
        columns=3,
        title="Counting combinations of all categorical features by Contraceptive method used (CMU)",
    )
    counts_cat_by_target.save(out_dir + "/counts_categorical_by_target.png")

if __name__ == "__main__":
  print(opt)
  main(opt["--train_dir"],  opt["--out_dir"])
