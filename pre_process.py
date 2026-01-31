"""
A script containing the Process class, primarily used for pre-processing the training data ready to pass to the model.
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

class Process:


    __slots__ = training_data


    def __init__(self, training_data: str) -> None:

        """

        Initialises the class with training data.

        :param training_data: A string of the filepath to the data CSV.

        """

        self.training_data = pd.read_csv(training_data)


    def separate_data(self) -> None:
        
        """

        Separates the data in "SalePrice" column from the rest of the training data, as the model has to not see this data. 

        Also prepares the "SalePrice" data as a Pandas Series against which to compare model predictions.

        """

        input_features = self.training_data.drop(
            ["SalePrice", "Id"],
            axis=1
        )
        sale_prices = self.training_data["SalePrice"].values

        return input_features, sale_prices


    # Identifies the column types for ease of use.

    num_cols = [c for c in input_features.columns if x[c].dtype != "object"]
    cat_cols = [c for c in input_features.columns if x[c].dtype == "object"]

    # Builds a pre-processing pipeline with sklearn.

    numeric_transformer = Pipeline(
        steps=[('scaler', StandardScaler())]
    )

    categorical_transformer = Pipeline(
        steps=[('encoder', OneHotEncoder(handle_unknown='ignore'))]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, num_cols),
            ('cat', categorical_transformer, cat_cols)
        ]
    )