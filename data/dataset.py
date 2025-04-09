import torch
from torch.utils.data import Dataset
import numpy as np
from typing import Tuple
from ..config import CATEGORICAL_FEATURES, NUMERIC_FEATURES

class VehicleDataset(Dataset):
    """Dataset cho dữ liệu xe hơi."""
    
    def __init__(self, dataframe):
        """
        Khởi tạo dataset.
        
        Args:
            dataframe (pd.DataFrame): DataFrame chứa dữ liệu đã được xử lý
        """
        self.data = dataframe.reset_index(drop=True)
        
    def __len__(self) -> int:
        """Trả về số lượng mẫu trong dataset."""
        return len(self.data)
        
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Lấy một mẫu từ dataset.
        
        Args:
            idx (int): Chỉ số của mẫu
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Cặp tensor chứa dữ liệu categorical và numeric
        """
        cat_data = self.data.loc[idx, CATEGORICAL_FEATURES].values.astype(np.int64)
        num_data = self.data.loc[idx, NUMERIC_FEATURES].values.astype(np.float32)
        return torch.tensor(cat_data), torch.tensor(num_data) 