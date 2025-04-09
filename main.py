import torch
from torch.utils.data import DataLoader
import logging
from config import DATA_PATH, MODEL_CONFIG, CATEGORICAL_FEATURES
from data.data_processor import DataProcessor
from data.dataset import VehicleDataset
from models.model import VehicleEmbeddingModel
from models.recommender import VehicleRecommender

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    try:
        # Khởi tạo data processor
        data_processor = DataProcessor(DATA_PATH)
        
        # Load và tiền xử lý dữ liệu
        df = data_processor.load_data()
        processed_df, label_encoders = data_processor.preprocess_data(df)
        
        # Tạo dataset và dataloader
        dataset = VehicleDataset(processed_df)
        dataloader = DataLoader(
            dataset, 
            batch_size=MODEL_CONFIG['batch_size'], 
            shuffle=False
        )
        
        # Khởi tạo mô hình
        cat_dims = [df[col].nunique() for col in CATEGORICAL_FEATURES]
        model = VehicleEmbeddingModel(cat_dims)
        
        # Khởi tạo recommender
        recommender = VehicleRecommender(model, data_processor)
        
        # Tạo embeddings
        recommender.embeddings = recommender.generate_embeddings(dataloader)
        
        # Thử gợi ý
        recommendations = recommender.recommend_similar(0, top_k=5)
        print("\nCác xe được gợi ý:")
        print(recommendations)
        
    except Exception as e:
        logger.error(f"Lỗi trong quá trình chạy: {str(e)}")
        raise

if __name__ == "__main__":
    main() 