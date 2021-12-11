
# Dockerfile for Conttraceptive Method Predictor
# author: Christopher Alexander, Harry Chan, Abhiket Gaurav,Valli A
# date: 2021-12-11

# Use jupyter/datascience-notebook as the base image
FROM jupyter/datascience-notebook

# Install required python and R packages
RUN conda install --quiet --yes \
    'shap' \
    'lightgbm' \
    'seaborn' \
    'altair' \
    "scikit-learn" \
    "requests>=2.24.0" \
    "pandas>=1.3.*" \
    'pip' \
    "matplotlib>=3.2.2" \
    'pandoc' \
    'docopt' \
    'altair_saver' \
    'r-reticulate' \
    'r-bookdown'
    
# Install bookdown packages using Rscript
RUN  R -e 'install.packages("bookdown",repos = "http://cran.us.r-project.org")'
