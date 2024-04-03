import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler

# Load data
def load_data(path: str = "sample_sales_data.csv"):
    df = pd.read_csv(path)
    df.drop(columns=["Unnamed: 0"], inplace=True, errors='ignore')
    return df

# Create target variable and predictor variables
def create_target_and_predictors(data: pd.DataFrame = None, target: str = "estimated_stock_pct"):
    if target not in data.columns:
        raise Exception(f"Target: {target} is not present in the data")
    X = data.drop(columns=[target])
    y = data[target]
    return X, y

# Train algorithm with cross-validation
def train_algorithm_with_cross_validation(X: pd.DataFrame = None, y: pd.Series = None):
    K = 10
    SPLIT = 0.75
    accuracy = []
    for fold in range(0, K):
        model = RandomForestRegressor()
        scaler = StandardScaler()
        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=SPLIT, random_state=42)
        scaler.fit(X_train)
        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)
        trained_model = model.fit(X_train, y_train)
        y_pred = trained_model.predict(X_test)
        mae = mean_absolute_error(y_true=y_test, y_pred=y_pred)
        accuracy.append(mae)
        print(f"Fold {fold + 1}: MAE = {mae:.3f}")
    print(f"Average MAE: {(sum(accuracy) / len(accuracy)):.2f}")

# Main function to execute training pipeline
def run():
    df = load_data()
    X, y = create_target_and_predictors(data=df)
    train_algorithm_with_cross_validation(X=X, y=y)

if __name__ == "__main__":
    run()
