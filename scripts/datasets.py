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
        self._dataset.dropna(inplace=True)


class CustomerDataset(DatasetBase):
    def __init__(self, dataset_path):
        super().__init__()
        df = pd.read_csv(dataset_path).drop(columns=["CustomerID"])
        self._dataset = df
        self._clean_dataframe()

    def _clean_dataframe(self):
        # Remove rows with null values
        self._dataset.dropna(inplace=True)

        # Remove rows with weird values
        unwanted_rows = []
        for idx in self._dataset.index:
            if self._dataset['Gender'][idx] not in ['Male', 'Female']:
                unwanted_rows.append(idx)
            elif self._dataset['Age'][idx] < 18:
                unwanted_rows.append(idx)
            elif self._dataset['Spending Score (1-100)'][idx] < 1 or self._dataset['Spending Score (1-100)'][idx] > 100:
                unwanted_rows.append(idx)
            elif self._dataset['Work Experience'][idx] < 0:
                unwanted_rows.append(idx)
            elif self._dataset['Family Size'][idx] <= 0:
                unwanted_rows.append(idx)

        self._dataset.drop(unwanted_rows, inplace=True)
        self._dataset.reset_index(inplace=True)


if __name__ == '__main__':
    md = MarketDataset(r"C:\Important Stuff\Facultate\Anul V\DM\Proiect\dmdw_project\data\marketing_campaign.csv")
    print(md.get_numpy())
    print(md.get_pandas())
    print(len(md))
