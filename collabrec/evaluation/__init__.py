# This package provides evaluation tools: train/test splits, metrics, runners, and dataset statistics.


from collabrec.evaluation.split import random_train_test_split, leave_one_out_split
from collabrec.evaluation.metrics import rmse, mae, coverage, EvaluationResult
from collabrec.evaluation.runner import evaluate_random_split, evaluate_leave_one_out
from collabrec.evaluation.stats import DatasetStats