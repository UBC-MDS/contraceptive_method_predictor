# Contraceptive Method Predictor
- author: Christopher Alexander
- contributors: Harry Chan, Abhiket Gaurav, Valli A

A data analysis project to predict the contraceptive method choice of a woman based on her demographic and socio-economic characteristics.
# About

Here we attempt to build a classification model which can use the demographic and socio-economic data to predict the choice of contraceptive method of a woman. The choices has been categorized into : 1=No-use, 2=Long-term, 3=Short-term. We have created historgrams to understand the data distribution (of both the target and predictor variables). We have also made correlation plots to understand, the relationship between variables.

# Usage
To replicate the analysis, clone this GitHub repository, install the depndencies listede below, and run the following command at the commandline / terminal from the root directory of this project:

    python src/download_data.py --url=https://archive.ics.uci.edu/ml/machine-learning-databases/cmc/cmc.data --out_file=data/raw/contraceptive.csv
    jupyter lab src/eda.ipynb


# Initial EDA
At this intial stage we have done an EDA of the dataset and have found that 

- Majority of the women(~43% in the given dataset) are not using any contraceptive method
- Majority of the observations are  of the 'wife age' between 22-36years
- Most of them have 2-3 children
- Women have high level of education, Husbands too have high level of education.
- Most of women are not working
- Women have high standard of living albeit with no media exposure

# Dependencies
- Python 3.7.3 and Python packages:
    - docopt==0.6.2
    - pandas==1.3.4
    - altair==4.1.0
    - numpy==1.21.4
    - scikit-learn==1.0.1

# License
The source code for the site is licensed under the MIT license, which you can find [here](https://github.com/UBC-MDS/contraceptive_method_predictor/blob/main/LICENSE).
# References 
Tjen-Sien Lim (limt '@' stat.wisc.edu) http://archive.ics.uci.edu/ml/datasets/Contraceptive+Method+Choice

Lim, T.-S., Loh, W.-Y. & Shih, Y.-S. (1999). A Comparison of Prediction Accuracy, Complexity, and Training Time of Thirty-three Old and New Classification Algorithms. Machine Learning. 
