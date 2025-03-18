import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from imblearn.under_sampling import RandomUnderSampler
import os

def preprocess_data(df):
    """
    Preprocess the data by performing undersampling and saving the processed data.
    
    Parameters:
    df (DataFrame): The input DataFrame to preprocess.
    
    Returns:
    DataFrame: The downsampled DataFrame.
    """
    # Separate features (X) and target (y)
    X = df.drop('Class', axis=1)
    y = df['Class']

    # Initialize RandomUnderSampler
    rus = RandomUnderSampler(random_state=42)

    # Fit and apply the resampler to the data
    X_resampled, y_resampled = rus.fit_resample(X, y)

    # Convert the resampled data back to a DataFrame
    downsampled_df = pd.concat([pd.DataFrame(X_resampled, columns=X.columns), pd.DataFrame(y_resampled, columns=['Class'])], axis=1)

    return downsampled_df

def save_processed_data(df, file_path):
    """
    Save the processed data to a CSV file.
    
    Parameters:
    df (DataFrame): The DataFrame to save.
    file_path (str): The path to save the CSV file.
    """
    df.to_csv(file_path, index=False)
    print(f"Processed data saved to {file_path}")

def plot_heatmap(df, file_path):
    """
    Create and save a heatmap of the correlation matrix of the DataFrame.
    
    Parameters:
    df (DataFrame): The input DataFrame to plot the heatmap.
    file_path (str): The path to save the heatmap image.
    """
    file_path = 'artifacts/heatmap.jpeg'
    plt.figure(figsize=(16, 9))
    sns.heatmap(df.corr(), annot=True)
    plt.savefig(file_path)
    print(f"Heatmap saved to {file_path}")

# Example usage:
# Load the data
data_path = 'data/raw_data/creditcard.csv'
df = pd.read_csv(data_path)

# Preprocess the data
downsampled_df = preprocess_data(df)

# Save the processed data
processed_data_path = 'data/processed/processed_data.csv'
save_processed_data(df, processed_data_path)

# Plot and save the heatmap
heatmap_path = 'artifacts/heatmap.jpeg'
plot_heatmap(downsampled_df, heatmap_path)
