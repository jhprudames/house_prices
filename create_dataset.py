"""

A module containing the CreateDataset class, used to create the dataset for
model. The class will utilise the outputs of the PreProcess class to
create the dataset.

"""

import numpy as np
import torch
from torch.utils.data import Dataset
from typing import List


class HousePriceDataset(Dataset):

    """

    A class used to convert the outputs of the PreProcess class into
    PyTorch tensors, ready for the model to ingest.

    """


    def __init__(self, features: np.ndarray, targets: np.ndarray) -> None:

        """

        Initialises the HousePriceDataset class.

        """

        self.processed_features = torch.tensor(
            features, dtype=torch.float32
        )

        self.sale_prices = torch.tensor(
            targets, dtype=torch.float32
        )    


    def __len__(self) -> int:

        """

        A required method as part of the Dataset class API.

        """

        return len(self.processed_features)


    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:

        """

        A required method as part of the Dataset class API. Allows PyTorc
        load mini-batches (ask for one row at a time).

        """

        return self.processed_features[idx], self.sale_prices[idx]
   

    @property
    def processed_features(self) -> torch.Tensor:

        """
        
        Returns the pre-processed features tensor.
        
        """

        return self.processed_features

    
    @property
    def sale_prices(self) -> torch.Tensor:

        """

        Returns the sale price tensor (the answers for the model).
        
        """
        return self.sale_prices
