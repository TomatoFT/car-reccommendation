from typing import List

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