from scripts.train_model import run_all_experiments, train_test_split, preprocess_data
from scripts.promote_model import promote_best_model_to_production

def main():
    # Train and log the model
    X, y = preprocess_data()

    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Run all experiments (train models and log them)
    run_all_experiments(X_train, X_test, y_train, y_test)
    
    # Promote the best model to production based on accuracy
    promote_best_model_to_production()

if __name__ == "__main__":
    main()
