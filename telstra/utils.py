import numpy as np
from sklearn.cross_validation import train_test_split


def evaluate(res, y_true):
    nrows = res.shape[0]
    res = res + 1e-15
    probs = res[list(range(nrows)), y_true]
    return - np.sum(np.log(probs)) / nrows


def get_score_evaluate(x, y, estimator, random_state=43):
    x_train, x_valid, y_train, y_valid = train_test_split(
        x, y, test_size=.33, random_state=random_state)
    estimator.fit(x_train, y_train)
    score_train = estimator.score(x_train, y_train)
    score_valid = estimator.score(x_valid, y_valid)
    evaluate_train = evaluate(estimator.predict_proba(x_train), y_train)
    evaluate_valid = evaluate(estimator.predict_proba(x_valid), y_valid)
    return score_train, score_valid, evaluate_train, evaluate_valid


train_event_start = 3
test_event_start = train_event_start - 1
train_feature_start = 56
train_event_end = train_feature_start
test_feature_start = train_feature_start - 1
test_event_end = test_feature_start
train_resource_start = 442
train_feature_end = train_resource_start
test_resource_start = train_resource_start - 1
test_feature_end = test_resource_start
train_severity_start = 452
train_resource_end = train_severity_start
test_severity_start = train_severity_start - 1
test_resource_end = test_severity_start
train_volume_start = 457
train_severity_end = train_volume_start
test_volume_start = train_volume_start - 1
test_severity_end = test_volume_start
train_end = 843
test_end = train_end - 1

ranges = [
    ('event', range(train_event_start, train_feature_start)),
    ('feature', range(train_feature_start, train_resource_start)), 
    ('resource', range(train_resource_start, train_severity_start)),
    ('severity', range(train_severity_start, train_volume_start)),
    ('volume', range(train_volume_start, train_end))]

def powerset(l):
    if not l: return [[]]
    return powerset(l[1:]) + [[l[0]] + x for x in powerset(l[1:])]

def list_sum(args):
    res = []
    for arg in args:
        res += arg
    return res

def string_sum(args):
    res = ''
    for arg in args:
        res += arg + ' '
    return res

full_ranges = powerset(ranges)