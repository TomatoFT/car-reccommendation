from typing import List, Dict

# Cấu hình đường dẫn
DATA_PATH = "content/tonkieulam-data_09042025.csv"

# Cấu hình features
CATEGORICAL_FEATURES: List[str] = [
    'LeadStatus', 'LapsedContract', 'Dealer', 'Vehicle', 'Model',
    'Derivative', 'Fuel Type', 'Transmission', 'Body Style',
    'ProductType', 'AssetType'
]

NUMERIC_FEATURES: List[str] = [
    'TrueEquity', 'Equity', 'Parity', 'PurchasePrice', 'Advance',
    'MonthlyPayment', 'Balloon', 'Mileage', 'ValuationAmount',
    'Settlement', 'Rate', 'Term', 'Doors', 'PaymentsLeft',
    'PaymentsMade', 'LastSeenDistanceTravelled'
]

# Cấu hình model
MODEL_CONFIG: Dict = {
    'embed_dim': 8,
    'hidden_dim': 128,
    'output_dim': 64,
    'batch_size': 64,
    'learning_rate': 0.001
} 