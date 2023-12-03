from copy import deepcopy

import numpy as np
from sklearn.cluster import OPTICS
from sklearn.metrics import pairwise_distances, silhouette_score
from sklearn.model_selection import GridSearchCV

from scripts.datasets import DATASETS_LIST, get_dataset_object


PARAM_GRID = {
        "corr_thrs": [0.5, 0.7, 0.95],
        "optics": {
            "min_samples": [2, 5, 7, 10, 15],
            "metric": ["euclidean", "cosine", "minkowski"]
        }
    }


def metric_performance(estimator, X):
    estimator.fit(X)
    labels = estimator.labels_

    cluster_distances = pairwise_distances(X)

    # Average intra-cluster distance scaled
    intra_cluster_distances = np.mean(
        cluster_distances[labels != -1][:, labels != -1]) / cluster_distances.max()

    try:
        silhouette = silhouette_score(X[labels != -1], labels[labels != -1])
    except ValueError:
        silhouette = 0  # avg score

    score = silhouette - intra_cluster_distances

    return score


def hyperparam_finetune(dataset):
    best_score = -np.inf
    best_params = {"corr_thr": None, "optics": None}
    for corr_thr in PARAM_GRID["corr_thrs"]:
        dataset_iter = deepcopy(dataset)
        dataset_iter.dimensionality_reduction(corr_thr=corr_thr)

        model = OPTICS()
        grid_search = GridSearchCV(model, PARAM_GRID["optics"], scoring=metric_performance, cv=5)
        grid_search.fit(dataset_iter.get_numpy())

        if best_score < grid_search.best_score_:
            best_score = grid_search.best_score_
            best_params["corr_thr"] = corr_thr
            best_params["optics"] = grid_search.best_params_

    return best_params, best_score


def main_run(run_type):
    if run_type == "FINE_TUNE":
        for dataset_name, path in DATASETS_LIST.items():
            dataset = get_dataset_object(dataset_name, path)
            best_params, best_score = hyperparam_finetune(dataset)
            print(best_params)
            print(best_score)
    else:
        for dataset_name, path in DATASETS_LIST.items():
            dataset = get_dataset_object(dataset_name, path)
            model = OPTICS(metric="euclidean", min_samples=5)
            dataset.dimensionality_reduction(corr_thr=0.5)

            model.fit(dataset.get_numpy())
            print(model.labels_)
            print(model.labels_.max())
            print(len(model.labels_[model.labels_ != -1]))


if __name__ == '__main__':
    main_run("RUN")
