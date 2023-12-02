from sklearn.cluster import OPTICS

from scripts.datasets import DATASETS_LIST, get_dataset_object


def main_run():
    for dataset_name, path in DATASETS_LIST.items():
        dataset = get_dataset_object(dataset_name, path)
        # TODO: use some type of cross validation (maybe GridSearch) for this parameter for both cases
        #  (not just MarketDataset) together with the parameters of optics (there are two).
        if dataset_name == "MarketDataset":
            dataset.dimensionality_reduction(corr_thr=0.5)

        clustering = OPTICS(min_samples=5).fit(dataset.get_numpy())
        print(clustering.labels_)


if __name__ == '__main__':
    main_run()
