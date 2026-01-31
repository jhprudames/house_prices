"""

A script containing the model which I will build and train on the training data in the data folder.

"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_log_error
from torch.utils.data import Dataset, DataLoader

class RegressionNN(nn.Module):

    """

    A simple neural network regression module, using nn.Module functionality.

    """

    def __init__(self, input_dim: int) -> None:

        """
        
        Initialises the model. The number of input dimensions can be changed
        depending on the length of the input data set.

        """

        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        """
        
        Completes a forward pass through the neural network.

        """

        return self.model(x)
