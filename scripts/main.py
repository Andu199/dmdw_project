from copy import deepcopy

import numpy as np
from matplotlib import pyplot as plt
from sklearn.cluster import OPTICS
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.model_selection import GridSearchCV

from scripts.datasets import DATASETS_LIST, get_dataset_object

HIGH_VALUE = 10000
PARAM_GRID = {
        "n_components": [3, 5, 7],
        "optics": {
            "min_samples": [7, 10, 15],
            "metric": ["euclidean", "cosine", "minkowski"],
            "xi": [0.1, 0.4, 0.7],
        }
    }


def metric_performance(estimator, X):
    estimator.fit(X)
    labels = estimator.labels_

    try:
        silhouette = silhouette_score(X[labels != -1], labels[labels != -1])
    except ValueError:
        silhouette = 0  # avg score

    clustered_points = len(X[labels != -1])
    noise_points = len(X[labels == -1])
    total_points = len(X)

    try:
        db_score = davies_bouldin_score(X, labels)
    except ValueError:
        db_score = HIGH_VALUE

    score = silhouette - db_score + (clustered_points / total_points) - (noise_points / total_points)

    return score


def hyperparam_finetune(dataset):
    best_score = -np.inf
    best_params = {"n_components": None, "optics": None}
    for n_components in PARAM_GRID["n_components"]:
        if n_components > dataset.get_pandas().shape[1]:
            continue

        dataset_iter = deepcopy(dataset)
        dataset_iter.dimensionality_reduction(n_components=n_components)

        model = OPTICS()
        grid_search = GridSearchCV(model, PARAM_GRID["optics"], scoring=metric_performance, cv=5)
        grid_search.fit(dataset_iter.get_numpy())

        if best_score < grid_search.best_score_:
            best_score = grid_search.best_score_
            best_params["n_components"] = n_components
            best_params["optics"] = grid_search.best_params_

    return best_params, best_score


def evaluate_and_visualize_best_result(best_params, dataset, dataset_name):
    optics = OPTICS(**best_params['optics'])
    train_dataset = deepcopy(dataset)
    train_dataset.dimensionality_reduction(n_components=best_params['n_components'])

    optics.fit(train_dataset.get_numpy())

    print(f"\n===Results for {dataset_name}===\n")
    print("Parameters used:")
    print(best_params)
    print("No of clusters: " + str(optics.labels_.max()))
    print("No of clustered points: " + str(len(optics.labels_[optics.labels_ != -1])))
    print("Silhouette Score: " + str(silhouette_score(
        train_dataset.get_numpy()[optics.labels_ != -1],
        optics.labels_[optics.labels_ != -1]
    )))

    print("Davies Bouldin Score: " + str(davies_bouldin_score(
        train_dataset.get_numpy()[optics.labels_ != -1],
        optics.labels_[optics.labels_ != -1]
    )))

    print("Calinski Harabasz Score: " + str(calinski_harabasz_score(
        train_dataset.get_numpy()[optics.labels_ != -1],
        optics.labels_[optics.labels_ != -1]
    )))

    # Visualize
    dataset.dimensionality_reduction(n_components=3)
    points = dataset.get_numpy()

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=list(optics.labels_))
    plt.savefig(f"../outputs/results_{dataset_name}.jpg")
    plt.clf()

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    idx_not_noise = np.where(optics.labels_ != -1)
    ax.scatter(points[idx_not_noise, 0], points[idx_not_noise, 1], points[idx_not_noise, 2],
               c=list(optics.labels_[idx_not_noise]))
    plt.savefig(f"../outputs/results_{dataset_name}_wo_noise.jpg")
    plt.clf()


def main_run(run_type):
    if run_type == "FINE_TUNE":
        for dataset_name, path in DATASETS_LIST.items():
            dataset = get_dataset_object(dataset_name, path)
            best_params, best_score = hyperparam_finetune(dataset)

            evaluate_and_visualize_best_result(best_params, dataset, dataset_name)
    elif run_type == "RUN":
        dataset = get_dataset_object("MarketDataset", "../data/marketing_campaign.csv")
        evaluate_and_visualize_best_result({
            'n_components': 3,
            'optics': {"min_samples": 15, "metric": "cosine", "xi": 0.1}}, dataset, "MarketDataset")

        dataset = get_dataset_object("CustomerDataset", "../data/Customers.csv")
        evaluate_and_visualize_best_result({
            'n_components': 5,
            'optics': {"min_samples": 7, "metric": "cosine", "xi": 0.1}}, dataset, "CustomerDataset")
    else:
        raise ValueError(f"Unknown run type! (Expected: FINE_TUNE or RUN; Got: {run_type})")


if __name__ == '__main__':
    np.random.seed(424242)
    main_run("RUN")
