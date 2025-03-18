import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
import pickle

def load_processed_data(file_path):
    """
    Load the processed data from a CSV file.
    
    Parameters:
    file_path (str): Path to the processed data file.
    
    Returns:
    DataFrame: Loaded DataFrame.
    """
    return pd.read_csv(file_path)

def prepare_training_data(df):
    """
    Prepare the training data by downsampling the majority class.
    
    Parameters:
    df (DataFrame): The input DataFrame with features and target.
    
    Returns:
    Tuple: Downsampled training features (X_train_downsampled), 
           downsampled training target (y_train_downsampled),
           original test features (X_test_orig),
           original test target (y_test_orig).
    """
    # Separate features and target
    X = df.drop(columns='Class')
    y = df['Class']

    # Split the original data into training and testing sets with stratification
    X_train_orig, X_test_orig, y_train_orig, y_test_orig = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    # Combine training features and target for downsampling
    train_data = X_train_orig.copy()
    train_data['Class'] = y_train_orig

    # Separate majority and minority classes
    majority_class = train_data[train_data['Class'] == 0]
    minority_class = train_data[train_data['Class'] == 1]

    # Downsample the majority class
    majority_downsampled = resample(majority_class, 
                                    replace=False, 
                                    n_samples=len(minority_class), 
                                    random_state=42)

    # Combine the downsampled majority class with the minority class
    downsampled_train_data = pd.concat([majority_downsampled, minority_class])

    # Prepare the downsampled training set
    X_train_downsampled = downsampled_train_data.drop(columns='Class')
    y_train_downsampled = downsampled_train_data['Class']

    return X_train_downsampled, y_train_downsampled, X_test_orig, y_test_orig

def train_logistic_regression(X_train, y_train):
    """
    Train a logistic regression model.
    
    Parameters:
    X_train (DataFrame): Training features.
    y_train (Series): Training target.
    
    Returns:
    LogisticRegression: Trained logistic regression model.
    """
    log_reg = LogisticRegression()
    log_reg.fit(X_train, y_train)
    return log_reg

def save_model(model, directory, filename):
    """
    Save the trained model to a pickle file.
    
    Parameters:
    model: Trained model to save.
    directory (str): Directory to save the model.
    filename (str): Name of the pickle file.
    """
    if not os.path.exists(directory):
        os.makedirs(directory)
    model_filepath = os.path.join(directory, filename)
    with open(model_filepath, 'wb') as file:
        pickle.dump(model, file)
    print(f"Model saved to {model_filepath}")

def save_classification_report(y_true, y_pred, file_path):
    """
    Save the classification report as an image.
    
    Parameters:
    y_true (Series): True target values.
    y_pred (Series): Predicted target values.
    file_path (str): Path to save the classification report image.
    """
    report = classification_report(y_true, y_pred, output_dict=True)
    report_df = pd.DataFrame(report).transpose()

    plt.figure(figsize=(10, 6))
    sns.heatmap(report_df.iloc[:-1, :-1], annot=True, cmap='Blues', fmt='.2f')
    plt.title('Classification Report')
    plt.savefig(file_path)
    print(f"Classification report saved to {file_path}")

def plot_confusion_matrix(y_true, y_pred):
    """
    Plot the confusion matrix.
    
    Parameters:
    y_true (Series): True target values.
    y_pred (Series): Predicted target values.
    """
    conf_matrix = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Non-Fraud', 'Fraud'], 
                yticklabels=['Non-Fraud', 'Fraud'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()

# Example usage:
# Load the processed data
processed_data_path = 'data/processed/processed_data.csv'
df = load_processed_data(processed_data_path)
print("Data Loaded!")

# Prepare the training and testing data
X_train_downsampled, y_train_downsampled, X_test_orig, y_test_orig = prepare_training_data(df)

# Train the logistic regression model
log_reg = train_logistic_regression(X_train_downsampled, y_train_downsampled)
print("Model Trained!")

# Predict on the original test data
y_pred = log_reg.predict(X_test_orig)

# Plot the confusion matrix
plot_confusion_matrix(y_test_orig, y_pred)

# Save the model
save_model(log_reg, 'models', 'logistic_regression_model.pkl')

# Save the classification report as an image
classification_report_path = 'artifacts/classification_report.jpeg'
save_classification_report(y_test_orig, y_pred, classification_report_path)
print("Report Saved!")
