import os.path

import pandas as pd


class DatasetBase:
    def __init__(self):
        self._dataset = None

    def __getitem__(self, item):
        if not isinstance(self._dataset, pd.DataFrame):
            raise ValueError("Initialize the dataset with a pandas DataFrame!")
        return self._dataset.iloc[item, :]

    def __len__(self):
        if not isinstance(self._dataset, pd.DataFrame):
            raise ValueError("Initialize the dataset with a pandas DataFrame!")
        return len(self._dataset.index)

    def get_pandas(self):
        if not isinstance(self._dataset, pd.DataFrame):
            raise ValueError("Initialize the dataset with a pandas DataFrame!")
        return self._dataset

    def get_numpy(self):
        if not isinstance(self._dataset, pd.DataFrame):
            raise ValueError("Initialize the dataset with a pandas DataFrame!")
        return self._dataset.to_numpy()

    def pretty_print(self, start, end):
        if not isinstance(self._dataset, pd.DataFrame):
            raise ValueError("Initialize the dataset with a pandas DataFrame!")
        print(self._dataset.iloc[start:end, :].to_markdown())


class MarketDataset(DatasetBase):
    def __init__(self, dataset_path):
        super().__init__()
        df = pd.read_csv(dataset_path, sep='\t').drop(columns=["ID"])
        self._dataset = df
        self._clean_dataframe()

    def _clean_dataframe(self):
        pass


class AutomobileDataset(DatasetBase):
    # Maybe get annotations for this dataset for metric evaluation
    def __init__(self, datasets_root_path, use_train, use_test):
        super().__init__()
        if not (use_train or use_test):
            raise ValueError("use_train and use_test arguments cannot be both False in the same time!")

        if use_train:
            df_train = pd.read_csv(os.path.join(datasets_root_path, "Train.csv"))\
                .drop(columns=["ID", "Var_1", "Segmentation"])
        else:
            df_train = pd.DataFrame()
        if use_test:
            df_test = pd.read_csv(os.path.join(datasets_root_path, "Test.csv")).drop(columns=["ID", "Var_1"])
        else:
            df_test = pd.DataFrame()

        df = pd.concat([df_train, df_test], ignore_index=True)
        self._dataset = df
        self._clean_dataframe()

    def _clean_dataframe(self):
        pass


if __name__ == '__main__':
    md = AutomobileDataset(r"C:\Important Stuff\Facultate\Anul V\DM\Proiect\dmdw_project\data\\", use_train=True, use_test=True)
    print(md.get_numpy())
    print(md.get_pandas())
    print(len(md))
