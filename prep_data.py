from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split
import pandas as pd

def load_data():
    # Load dataset
    dataset = fetch_ucirepo(id=15)
    
    X = dataset.data.features.copy()
    y = dataset.data.targets.copy()

    # Combine into one DataFrame to drop rows with any missing values
    df = pd.concat([X, y], axis=1)
    df = df.dropna()  # Drop rows with NaN in either features or target

    # Convert target column to numeric
    df['Class'] = df['Class'].replace({'benign': 2, 'malignant': 4})

    # Separate clean data
    X_clean = df.drop('Class', axis=1)
    y_clean = df['Class']

    # Final check: Ensure everything is numeric
    X_clean = X_clean.apply(pd.to_numeric)
    y_clean = y_clean.astype(int)

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X_clean, y_clean, test_size=0.25, random_state=42
    )

    return X_train, X_test, y_train, y_test
