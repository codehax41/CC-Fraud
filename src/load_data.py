import pandas as pd
import os

def load_data(file_name):
    data_path = os.path.join('data', 'raw_data', file_name)
    
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"The file at {data_path} does not exist.")
    
    data = pd.read_csv(data_path)
    return data

# Example usage:
data = load_data('creditcard.csv')