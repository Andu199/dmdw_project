import numpy as np
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
        self.EDUCATION_TO_INT = {"Basic": 0, "2n Cycle": 1, "Graduation": 2, "Master": 3, "PhD": 4}
        self.INT_TO_EDUCATION = {0: "Basic", 1: "2n Cycle", 2: "Graduation", 3: "Master", 4: "PhD"}

        df = pd.read_csv(dataset_path, sep='\t').drop(columns=["ID"])
        self._dataset = df
        self._clean_dataframe()

    def _clean_dataframe(self):
        # Remove rows with null values
        self._dataset.dropna(inplace=True)

        # From Year to Age
        today_year = pd.to_datetime("today").year
        ages = []
        for idx in self._dataset.index:
            ages.append(today_year - self._dataset["Year_Birth"][idx])
        self._dataset["Age"] = pd.to_numeric(ages)
        self._dataset.drop("Year_Birth", axis="columns", inplace=True)

        # From Education to quantitative value
        self._dataset["Education_Value"] = pd.to_numeric([self.EDUCATION_TO_INT[education]
                                                          for education in self._dataset["Education"].tolist()])
        self._dataset.drop("Education", axis="columns", inplace=True)

        # From Dt_Customer to days as customer (compared to the last customer by join date)
        self._dataset["Dt_Customer"] = pd.to_datetime(self._dataset["Dt_Customer"], dayfirst=True)

        today = pd.to_datetime("today")
        days = []
        for idx in self._dataset.index:
            days.append((today - self._dataset["Dt_Customer"][idx]).days)

        days = np.array(days)
        days -= min(days)
        self._dataset["Days_Customer"] = pd.to_numeric(days)
        self._dataset.drop("Dt_Customer", axis="columns", inplace=True)

        # Merge Kidhome + Teenhome
        self._dataset["Children"] = self._dataset["Kidhome"] + self._dataset["Teenhome"]

        # Total Spending
        self._dataset["Total_Spending"] = self._dataset["MntWines"] + \
                                          self._dataset["MntFruits"] + \
                                          self._dataset["MntMeatProducts"] + \
                                          self._dataset["MntFishProducts"] + \
                                          self._dataset["MntSweetProducts"] + \
                                          self._dataset["MntGoldProds"]


class CustomerDataset(DatasetBase):
    def __init__(self, dataset_path):
        super().__init__()
        df = pd.read_csv(dataset_path).drop(columns=["CustomerID"])
        self._dataset = df
        self._clean_dataframe()

    def _clean_dataframe(self):
        # Remove rows with null values
        self._dataset.dropna(inplace=True)

        # Remove rows with unrealistic values
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
