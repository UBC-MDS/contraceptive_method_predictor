# author: Abhiket Gaurav
# date: 2021-11-25

"""Reads data csv data from path and stores partitioned data to a local filepath as a csv.
Usage: split_data.py --path=<path> --out_file=<out_file> 
 
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

opt = docopt(__doc__)


def main(path, out_file):
    data = pd.read_csv(path, header=0, names=columns)
    train_df, test_df = train_test_split(data, test_size=0.3, random_state=123) 
    try:
        train_df.to_csv(out_file+"train.csv", index=False)
        test_df.to_csv(out_file+"test.csv", index=False)
    except:
        os.makedirs(os.path.dirname(out_file))
        train_df.to_csv(out_file+"train.csv", index=False)
        test_df.to_csv(out_file+"test.csv", index=False)


if __name__ == "__main__":
    main(opt["--path"], opt["--out_file"])


