import matplotlib.pyplot as plt
import seaborn as sns

from scripts.datasets import DATASETS_LIST, get_dataset_object


def main_eda():
    for dataset_name, path in DATASETS_LIST.items():
        dataset = get_dataset_object(dataset_name, path)
        correlation_matrix = dataset.get_pandas().corr()
        sns.heatmap(correlation_matrix)
        plt.title(f"{dataset_name} Correlation Matrix")
        plt.tight_layout()
        plt.xticks(rotation=45)
        plt.savefig(f"../outputs/{dataset_name}_correlation_matrix.jpg")
        plt.clf()

    dataset = get_dataset_object("MarketDataset", "../data/marketing_campaign.csv")
    dataset.dimensionality_reduction(corr_thr=0.5)
    correlation_matrix = dataset.get_pandas().corr()
    sns.heatmap(correlation_matrix)
    plt.title("MarketDataset Correlation Matrix (after reducing dimensionality)")
    plt.tight_layout()
    plt.xticks(rotation=45)
    plt.savefig("../outputs/MarketDataset_correlation_matrix_reduced.jpg")
    plt.clf()


if __name__ == '__main__':
    main_eda()
