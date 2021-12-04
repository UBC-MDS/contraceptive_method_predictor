# author: Harry Chan, Christopher Alexander
# date: 2021-12-01

all: results/val_score_results.csv results/models/final_svc.pkl results/counts_categorical_by_target.png \
results/cm.png results/pr_curve.png results/roc_curve.png \
doc/contraceptive_method_predictor_report.html doc/contraceptive_method_predictor_report.md

# download data
data/raw/contraceptive.csv: src/download_data.py
	python src/download_data.py --url=https://archive.ics.uci.edu/ml/machine-learning-databases/cmc/cmc.data --out_file=data/raw/contraceptive.csv

# split into train & test
data/processed/train.csv data/processed/test.csv: src/split_data.py data/raw/contraceptive.csv
	python src/split_data.py --path=data/raw/contraceptive.csv --out_file=data/processed/

# create EDA figure and write to file
results/counts_categorical_by_target.png: src/eda.py data/processed/train.csv
	python src/eda.py --train_dir=data/processed/train.csv --out_dir=results

# pre-process data and tune model (here, we find best model and perform RandomSearchCV for best model in our case SVC)
results/val_score_results.csv results/models/final_svc.pkl: src/preprocess_model_selection.py data/processed/train.csv
	python src/preprocess_model_selection.py --path=data/processed/train.csv --score_file=results/val_score_results.csv --model_path=results/models/final_svc.pkl

# test model on unseen data
results/cm.png results/pr_curve.png results/roc_curve.png: src/predict.py results/models/final_svc.pkl data/processed/test.csv
	python src/predict.py --test_path=data/processed/test.csv --model=results/models/final_svc.pkl --output_path=results/

# render report
doc/contraceptive_method_predictor_report.html doc/contraceptive_method_predictor_report.md: doc/contraceptive_method_predictor_report.Rmd \
doc/cmp_refs.bib doc/01_Summary.Rmd doc/02_Intro.Rmd doc/03_Data.Rmd \
doc/04_EDA.Rmd doc/05_Method.Rmd doc/06_Results.Rmd doc/07_Acknowledgment.Rmd
	Rscript -e "rmarkdown::render('doc/contraceptive_method_predictor_report.Rmd',output_format = 'all')"

clean: 
	rm -rf data
	rm -rf results
	rm -rf doc/contraceptive_method_predictor_report.html doc/contraceptive_method_predictor_report.md
			