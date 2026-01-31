"""

A script containing the model which I will build and train on the training data in the data folder.

"""

import torch
import torch.nn as nn


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
