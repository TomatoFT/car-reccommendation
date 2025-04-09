import torch
import pandas as pd
from typing import List, Dict
import logging
from model import VehicleEmbeddingModel
from data.data_processor import DataProcessor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VehicleRecommender:
    """Hệ thống gợi ý xe hơi."""
    
    def __init__(self, model: VehicleEmbeddingModel, data_processor: DataProcessor):
        """
        Khởi tạo recommender.
        
        Args:
            model (VehicleEmbeddingModel): Mô hình embedding
            data_processor (DataProcessor): Bộ xử lý dữ liệu
        """
        self.model = model
        self.data_processor = data_processor
        self.model.eval()
        
    def generate_embeddings(self, dataloader: torch.utils.data.DataLoader) -> torch.Tensor:
        """
        Tạo embeddings cho toàn bộ dataset.
        
        Args:
            dataloader (DataLoader): DataLoader chứa dữ liệu
            
        Returns:
            torch.Tensor: Tensor chứa embeddings
        """
        all_embeddings = []
        with torch.no_grad():
            for cat_batch, num_batch in dataloader:
                embeddings = self.model(cat_batch, num_batch)
                all_embeddings.append(embeddings)
        return torch.cat(all_embeddings)
        
    def recommend_similar(self, vehicle_idx: int, top_k: int = 5) -> pd.DataFrame:
        """
        Gợi ý các xe tương tự.
        
        Args:
            vehicle_idx (int): Chỉ số của xe cần gợi ý
            top_k (int): Số lượng xe gợi ý
            
        Returns:
            pd.DataFrame: DataFrame chứa các xe được gợi ý
        """
        try:
            # Lấy embedding của xe cần gợi ý
            query = self.embeddings[vehicle_idx]
            
            # Tính cosine similarity
            similarities = torch.nn.functional.cosine_similarity(
                query.unsqueeze(0), 
                self.embeddings
            )
            
            # Lấy top-k indices (bỏ qua chính nó)
            top_indices = similarities.topk(top_k + 1).indices[1:]
            
            # Lấy thông tin các xe được gợi ý
            recommended_vehicles = self.data_processor.df.iloc[top_indices.numpy()]
            
            # Giải mã các features categorical
            decoded_recommendations = self.data_processor.decode_categorical(recommended_vehicles)
            
            logger.info(f"Đã tìm thấy {top_k} xe tương tự")
            return decoded_recommendations
            
        except Exception as e:
            logger.error(f"Lỗi khi gợi ý xe: {str(e)}")
            raise 