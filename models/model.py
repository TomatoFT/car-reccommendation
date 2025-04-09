import torch
import torch.nn as nn
from typing import List
from ..config import MODEL_CONFIG

class VehicleEmbeddingModel(nn.Module):
    """Mô hình embedding cho xe hơi."""
    
    def __init__(self, cat_dims: List[int]):
        """
        Khởi tạo mô hình.
        
        Args:
            cat_dims (List[int]): Danh sách số lượng categories cho mỗi feature categorical
        """
        super().__init__()
        
        # Cấu hình từ file config
        embed_dim = MODEL_CONFIG['embed_dim']
        hidden_dim = MODEL_CONFIG['hidden_dim']
        output_dim = MODEL_CONFIG['output_dim']
        
        # Embedding layers cho categorical features
        self.cat_embeddings = nn.ModuleList([
            nn.Embedding(num_categories, embed_dim) for num_categories in cat_dims
        ])
        
        # Tính toán tổng số chiều của embeddings
        self.embed_dim_total = embed_dim * len(cat_dims)
        
        # Fully connected layers
        self.fc = nn.Sequential(
            nn.Linear(self.embed_dim_total + len(MODEL_CONFIG['NUMERIC_FEATURES']), hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.ReLU()
        )
        
    def forward(self, cat_input: torch.Tensor, num_input: torch.Tensor) -> torch.Tensor:
        """
        Forward pass của mô hình.
        
        Args:
            cat_input (torch.Tensor): Input tensor cho categorical features
            num_input (torch.Tensor): Input tensor cho numeric features
            
        Returns:
            torch.Tensor: Output tensor
        """
        # Embed categorical features
        cat_embeds = [embed(cat_input[:, i]) for i, embed in enumerate(self.cat_embeddings)]
        cat_embeds = torch.cat(cat_embeds, dim=1)
        
        # Concatenate với numeric features
        x = torch.cat([cat_embeds, num_input], dim=1)
        
        # Pass qua fully connected layers
        x = self.fc(x)
        return x 