import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import os

# Enable MLflow autologging
mlflow.sklearn.autolog()

def main():
    # Load data
    df = pd.read_csv('titanic_preprocessing/titanic_clean.csv')
    
    # Feature selection
    X = df.drop('Survived', axis=1)
    y = df['Survived']
    
    # Train test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Start MLflow run
    with mlflow.start_run(run_name="CI_RandomForest"):
        # Initialize and train model
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        # Predictions
        y_pred = model.predict(X_test)
        
        # Calculate accuracy
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Accuracy: {accuracy:.4f}")

if __name__ == "__main__":
    main()
