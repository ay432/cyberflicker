from sklearn import datasets
from skopt import BayesSearchCV
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold
import pandas as pd
import numpy as np

# Load Iris dataset
iris = datasets.load_iris()
X = iris.data
y = iris.target

# Define the estimator
estimator = xgb.XGBClassifier(
    n_jobs=-1,
    objective='multi:softmax',
    eval_metric='merror',
    verbosity=0,
    num_class=len(set(y))
)

# Define the search space
search_space = {
    'learning_rate': (0.01, 1.0, 'log-uniform'),
    'min_child_weight': (0, 10),
    'max_depth': (1, 50),
    'max_delta_step': (0, 10),
    'subsample': (0.01, 1.0, 'uniform'),
    'colsample_bytree': (0.01, 1.0, 'log-uniform'),
    'colsample_bylevel': (0.01, 1.0, 'log-uniform'),
    'reg_lambda': (1e-9, 1000, 'log-uniform'),
    'reg_alpha': (1e-9, 1.0, 'log-uniform'),
    'gamma': (1e-9, 8.5, 'log-uniform'),
    'n_estimators': (5, 5000),
    'scale_pos_weight': (1e-6, 500, 'log-uniform')
}

# Stratified cross-validation
crossvalidation = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# BayesSearchCV object
bayes_cv_tuner = BayesSearchCV(
    estimator=estimator,
    search_spaces=search_space,
    scoring='accuracy',
    cv=crossvalidation,
    n_jobs=-1,
    verbose=0,
    refit=True
)

# Callback to print and log status
def print_status(optim_results):
    # Ensure optimizer_results_ is non-empty
    if len(bayes_cv_tuner.optimizer_results_) > 0:
        best_index = np.argmax([1 - r.fun for r in bayes_cv_tuner.optimizer_results_ if r.fun is not None])
        best_result = bayes_cv_tuner.optimizer_results_[best_index]
        print(
            f"Iteration: {len(bayes_cv_tuner.optimizer_results_)}\n"
            f"Best score so far: {1 - best_result.fun:.4f}\n"
            f"Best parameters so far: {best_result.x}\n"
        )
        # Save results to CSV
        pd.DataFrame(
            [{**r.x, "score": 1 - r.fun} for r in bayes_cv_tuner.optimizer_results_ if r.fun is not None]
        ).to_csv("xgb_optimizer_results.csv", index=False)

# Fit the model
results = bayes_cv_tuner.fit(X, y, callback=print_status)

# Print final results after fitting
print(f"Best accuracy: {bayes_cv_tuner.best_score_}")
print(f"Best parameters: {bayes_cv_tuner.best_params_}")

#Best accuracy: 0.9600000000000002
#Best parameters: OrderedDict({'colsample_bylevel': 0.4597498022454739, 'colsample_bytree': 0.37406892381358803,
# 'gamma': 0.00020240789114048236, 'learning_rate': 0.01, 'max_delta_step': 0, 'max_depth': 1, 'min_child_weight': 0,
# 'n_estimators': 1101, 'reg_alpha': 1.7781127174166097e-05, 'reg_lambda': 1e-09, 'scale_pos_weight': 257.9873634115973,
# 'subsample': 0.5936682764114035})
