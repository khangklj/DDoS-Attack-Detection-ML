import os
import argparse
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier  # Model

SEPERATOR = "================="

def preprocess(df):    
    """
    Preprocesses the given dataframe by performing the following operations:
    1. Drops the unnecessary columns specified in drop_columns list.
    2. Removes any rows containing inf or NaN values.
    3. Removes any duplicate rows.
    4. Encodes the 'Class' column into binary labels.
    5. Encodes the 'Label' column into integer labels.

    Parameters:
    - df (pandas.DataFrame): The dataframe to be preprocessed.

    Returns:
    - binary_le_mapping (dict): A dictionary mapping the original 'Class' labels to their binary labels.
    - multi_le_mapping (dict): A dictionary mapping the original 'Label' labels to their integer labels.
    """
    drop_columns = [ # drop unnecessary columns
    "Unnamed: 0",
    ]
    print("Drop columns: ", drop_columns)
    df.columns = df.columns.str.strip()
    df.drop(columns=drop_columns, inplace=True)
    print("Data shape: ", df.shape)

    print(SEPERATOR)

    # Remove inf and NaN
    print("Drop NaN: ")
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    print("NaN: ", df.isna().sum())
    df.dropna(inplace=True)
    print("Data shape: ", df.shape)

    print(SEPERATOR)

    # Remove duplicated
    print("Duplicated: ", df.duplicated().sum())
    df.drop_duplicates(inplace=True)
    df.reset_index(inplace=True, drop=True)
    print("Data shape: ", df.shape)

    print(SEPERATOR)

    # Encode Class column to 0 or 1
    print("Encode Class column")
    binary_le = LabelEncoder()
    df['Class'] = binary_le.fit_transform(df['Class'])
    binary_le_mapping = dict(zip(binary_le.classes_, binary_le.transform(binary_le.classes_)))
    for key, value in binary_le_mapping.items():
        print(f"{key}: {value}")
    print("Data shape: ", df.shape)


    print(SEPERATOR)

    # Encode Label column to 0,1,2,... for each unique value of df
    print("Encode Label column")
    multi_le = LabelEncoder()
    df['Label'] = multi_le.fit_transform(df['Label'])
    multi_le_mapping = dict(zip(multi_le.classes_, multi_le.transform(multi_le.classes_)))
    for key, value in multi_le_mapping.items():
        print(f"{key}: {value}")
    print("Data shape: ", df.shape)

    return binary_le_mapping, multi_le_mapping
  
def binary_model(df, binary_le_mapping):
    """
    Train a Random Forest classifier on the given dataframe and evaluate its performance. 
    
    Parameters:
    - df (pandas.DataFrame): The dataframe containing the data.
    - binary_le_mapping (dict): A dictionary mapping the binary labels to their corresponding integer labels.
    
    Returns:
    - y_test (pandas.Series): The true binary labels.
    - y_pred (pandas.Series): The predicted binary labels.
    """
    # Selecting features except for 'Label' and 'Class'
    X = df.drop(labels=['Label','Class'], axis=1)
    
    # Selecting the 'Class' column
    y = df['Class']
    
    # Splitting the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Creating a Random Forest classifier
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    
    # Training the model
    model.fit(X_train, y_train)
    
    # Making predictions on the test set
    y_pred = model.predict(X_test)
    
    # Evaluating the model's performance
    accuracy = accuracy_score(y_test, y_pred) # Accuracy
    precision = precision_score(y_test, y_pred) # Precision
    recall = recall_score(y_test, y_pred) # Recall
    f1 = f1_score(y_test, y_pred) # F1-score
    
    # Printing the evaluation metrics
    print("Accuracy", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1-score:", f1)
    
    return y_test, y_pred

def multi_model(df, multi_le_mapping):    
    """
    Train a Random Forest classifier on the given dataframe and evaluate its performance. 
    
    Parameters:
    - df (pandas.DataFrame): The dataframe containing the data.
    - multi_le_mapping (dict): A dictionary mapping the multi-class labels to their corresponding integer labels.
    
    Returns:
    - y_test (pandas.Series): The true labels.
    - y_pred (pandas.Series): The predicted labels.
    """

    # Selecting features except for 'Label' and 'Class'
    X = df.drop(labels=['Label','Class'], axis=1)
    
    # Selecting the 'Label' column
    y = df['Label']

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create a Random Forest classifier
    model = RandomForestClassifier(n_estimators=100, random_state=42)

    # Train the model
    model.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred = model.predict(X_test)

    # Evaluate the model's performance
    accuracy = accuracy_score(y_test, y_pred) # Accuracy
    precision = precision_score(y_test, y_pred, average="weighted") # Precision
    recall = recall_score(y_test, y_pred, average="weighted") # Recall
    f1 = f1_score(y_test, y_pred, average="weighted") # F1-score

    # Printing the evaluation metrics
    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1-score:", f1)

    return y_test, y_pred


def visualization(y_test, y_pred, mapping, is_saved_fig=False):
    """
    Visualize the performance of a classifier by plotting a confusion matrix and a bar plot of accuracy and misclassification rate.
    
    Parameters:
    y_test (pandas.Series): The true labels.
    y_pred (pandas.Series): The predicted labels.
    mapping (dict): A dictionary mapping the labels to their corresponding integer labels.
    is_saved_fig (bool, optional): Whether to save the confusion matrix figure and model performance figure. Default is False.
    
    Returns:
    None
    """
    
    # Create a Log folder if not exists
    if is_saved_fig:
        # Define the directory path
        log_dir = os.path.join(os.getcwd(), "log")
        
        # Create the directory if it does not exist
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
            
    # Get the labels from the mapping
    labels = list(mapping.keys())

    num_classes = len(mapping)
    if (num_classes == 2):
        prefix_title = "binary"
    else:
        prefix_title = "multi"
    
    # Create a confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    
    # Plot the confusion matrix
    plt.figure(figsize=(16, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')

    # Save the confusion matrix figure if save_fig_path is not None
    if is_saved_fig:
        # Define the file path
        file_path = os.path.join(log_dir, f'${prefix_title}_confusion_matrix.png')
        
        # Save the figure
        plt.savefig(file_path)

    plt.show()   
        
    # Calculate accuracy and misclassification rate
    accuracy = accuracy_score(y_test, y_pred)
    misclassification_rate = 1 - accuracy
    
    # Create a bar plot for accuracy and misclassification rate
    plt.figure(figsize=(6, 4))
    plt.bar(['Accuracy', 'Misclassification Rate'], [accuracy, misclassification_rate])
    plt.ylim(0, 1)
    plt.title('Model Performance')
    plt.ylabel('Rate')

    # Save the model performance figure if save_fig_path is not None
    if is_saved_fig:
        # Define the file path
        file_path = os.path.join(log_dir, f'${prefix_title}_model_performance.png')
        
        # Save the figure
        plt.savefig(file_path)

    plt.show()
    
    

def main():
    parser = argparse.ArgumentParser(description='DDoS Attack Detection')
    parser.add_argument('--classification_type', '-c', type=str, default='both', help='Classification type. Default is "both". Options: "binary", "multi", "both"')
    parser.add_argument('--is_saved_fig', type=bool, default=False, help='Whether to save the confusion matrix figure. Default is False.')

    args = parser.parse_args()

    # Read data
    df = pd.read_csv('./dataset/cicddos2019_dataset.csv')
    print("BEFORE PREPROCESSING:")
    print("Data shape: ", df.shape)
    print(df)

    print(SEPERATOR)    

    # Preprocessing
    print("PREPROCESSING:")
    binary_le_mapping, multi_le_mapping = preprocess(df)
    
    print(SEPERATOR)

    print("AFTER PREPROCESSING:")   
    print(df.shape)
    print(df)

    print(SEPERATOR)
    
    if args.classification_type == 'binary':
        print("Train and test model (binary_case):")
        y_test, y_pred = binary_model(df, binary_le_mapping)
        visualization(y_test, y_pred, binary_le_mapping, args.is_saved_fig)
            
    elif args.classification_type == 'multi':
        print("Train and test model (multi_case):")
        y_test, y_pred = multi_model(df, multi_le_mapping)
        visualization(y_test, y_pred, multi_le_mapping, args.is_saved_fig)

    elif args.classification_type == 'both':
        print("Train and test model (both_case):")
        y_test, y_pred = binary_model(df, binary_le_mapping)
        visualization(y_test, y_pred, binary_le_mapping, args.is_saved_fig)
        y_test_multi, y_pred_multi = multi_model(df, multi_le_mapping)
        visualization(y_test_multi, y_pred_multi, multi_le_mapping, args.is_saved_fig)

    print(SEPERATOR)
    print("END!")

if __name__ == '__main__':
    main()