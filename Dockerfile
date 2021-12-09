FROM jupyter/datascience-notebook
# RUN apt update
# RUN apt -y upgrade
# RUN apt-get install make
# RUN apt -y install r-base
# RUN apt -y install r-base-dev
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
    'r-reticulate'
RUN  R -e 'install.packages("bookdown",repos = "http://cran.us.r-project.org")'
# RUN  R -e 'install.packages("reticulate")'
# RUN conda install -c conda-forge/label/cf202003 altair_saver -y
