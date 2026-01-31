"""
A script containing the PreProcess class, primarily used for pre-processing the 
training data ready to pass to the model.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from typing import List

class PreProcess:


    __slots__ = training_data


    def __init__(self, training_data: str) -> None:

        """

        Initialises the class with training data.

        :param training_data: A string of the filepath to the data CSV.

        """

        self.training_data = pd.read_csv(training_data)


    def separate_training_data(self) -> None:
        
        """

        Separates the data in "SalePrice" column from the rest of the training 
        data, as the model has to not see this data. 

        Also prepares the "SalePrice" data as a Pandas Series against which to 
        compare model predictions .

        """

        input_features = self.training_data.drop(
            ["SalePrice", "Id"],
            axis=1
        )
        sale_prices = self.training_data["SalePrice"].values

        return input_features, sale_prices


    def separate_nums_and_categories(
        self, input_features: DataFrame
    ) -> List:

        """
        
        Further separates the input features in the training data based on 
        whether or not they contain numbers.

        The model will only be able to ingest numbers, so this step is crucial.

        """
        
        num_cols = [
            col for ccol in input_features.columns
            if input_features[col].dtype != "object"
        ]

        category_cols = [
            col for col in input_features.columns
            if input_features[col].dtype == "object"
        ]

        return num_cols, category_cols


    def preprocess_data(self, num_cols, category_cols) -> np.array:

        """
        
        Pre-processes the data by scaling the numerical values and converting
        the categorical entries into numbers with one-hot encoding.

        Returns the combined pre-processed data as one object ready to be
        passed to the model.

        """
        
        numeric_transformer = Pipeline(
            steps=[('scaler', StandardScaler())]
        )
        
        categorical_transformer = Pipeline(
            steps=[('encoder', OneHotEncoder(handle_unknown='ignore'))]
        )
        
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, num_cols),
                ('cat', categorical_transformer, category_cols)
            ]
        )

        return preprocessor


    def pipeline(self) -> List:

        """

        Ties other methods together into single callable method. User will only 
        use this method to return both pre-processed training data and sale 
        prices (which consistute the answers against which model predictions 
        can be checked).

        """

        input_features, sale_prices = self.separate_training_data()
        num_cols, category_cols = self.separate_nums_and_categories(
            input_features
        )
        preprocessed_data = self.preprocess_data(
            num_cols, category_cols
        )

        return preprocessed_data, sale_prices
