import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from typing import Tuple, Dict
import logging
from ..config import CATEGORICAL_FEATURES, NUMERIC_FEATURES

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataProcessor:
    """Class để xử lý và chuẩn bị dữ liệu cho mô hình."""
    
    def __init__(self, data_path: str):
        """
        Khởi tạo DataProcessor.
        
        Args:
            data_path (str): Đường dẫn đến file dữ liệu
        """
        self.data_path = data_path
        self.label_encoders: Dict[str, LabelEncoder] = {}
        self.scaler = StandardScaler()
        
    def load_data(self) -> pd.DataFrame:
        """Load và làm sạch dữ liệu."""
        try:
            df = pd.read_csv(self.data_path)
            df.columns = df.columns.str.strip()
            logger.info("Dữ liệu đã được load thành công")
            return df
        except Exception as e:
            logger.error(f"Lỗi khi load dữ liệu: {str(e)}")
            raise
            
    def preprocess_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, LabelEncoder]]:
        """
        Tiền xử lý dữ liệu.
        
        Args:
            df (pd.DataFrame): DataFrame chứa dữ liệu thô
            
        Returns:
            Tuple[pd.DataFrame, Dict[str, LabelEncoder]]: DataFrame đã được xử lý và các label encoders
        """
        try:
            # Chọn các features cần thiết
            selected_features = CATEGORICAL_FEATURES + NUMERIC_FEATURES
            df = df[selected_features].dropna()
            
            # Mã hóa categorical features
            for col in CATEGORICAL_FEATURES:
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col])
                self.label_encoders[col] = le
                
            # Chuẩn hóa numeric features
            df[NUMERIC_FEATURES] = self.scaler.fit_transform(df[NUMERIC_FEATURES])
            
            logger.info("Dữ liệu đã được tiền xử lý thành công")
            return df, self.label_encoders
            
        except Exception as e:
            logger.error(f"Lỗi khi tiền xử lý dữ liệu: {str(e)}")
            raise
            
    def decode_categorical(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Giải mã các features categorical.
        
        Args:
            df (pd.DataFrame): DataFrame chứa dữ liệu đã được mã hóa
            
        Returns:
            pd.DataFrame: DataFrame với các features đã được giải mã
        """
        decoded_df = df.copy()
        for col in CATEGORICAL_FEATURES:
            if col in self.label_encoders:
                decoded_df[col] = self.label_encoders[col].inverse_transform(df[col].values)
        return decoded_df 