from abc import ABC, abstractmethod
import pandas as pd

class BaseModel(ABC):

    name: str = "base_model"

    @abstractmethod
    def predict(self, input_data: pd.DataFrame) -> pd.DataFrame:
        """Make predictions on the input data.

        Args:
            input_data (pd.DataFrame): Input data for prediction.

        Returns:
            pd.DataFrame: DataFrame containing predictions.
        """
        raise NotImplementedError