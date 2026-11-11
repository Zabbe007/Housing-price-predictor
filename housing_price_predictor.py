import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

'''Housing Price Predictor:
This program loads a dataset of house sizes (in square feet) and prices, explores the data, trains a linear regression model,
visualizes the results, and allows the user to predict house prices based on size input when prompted.
Dataset: housing_data_500.csv (contains 'Size' and 'Price' columns)
'''

# Load dataset
def load_data(file_path):
    try:
        df = pd.read_csv(file_path)
        if 'Size' not in df.columns or 'Price' not in df.columns:
            raise ValueError("CSV must contain 'Size' and 'Price' columns.")
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        return None


# Explore and visualize dataset
def explore_data(df):
    print("\n--- Basic Statistics ---")
    print(df.describe())

    plt.figure()
    plt.scatter(df['Size'], df['Price'])
    plt.xlabel("Size (sqft)")
    plt.ylabel("Price ($)")
    plt.title("House Size vs Price")
    plt.show()


# Train linear regression model
def train_model(df):
    X = df[['Size']]  # Feature
    y = df['Price']   # Target

    model = LinearRegression()
    model.fit(X, y)

    predictions = model.predict(X)
    r2 = r2_score(y, predictions)

    print("\n--- Model Details ---")
    print(f"Coefficient (slope): {model.coef_[0]:.4f}")
    print(f"Intercept: {model.intercept_:.4f}")
    print(f"R^2 Score: {r2:.4f}")

    return model


# Visualize model, scatter plot with regression line
def visualize_model(df, model):
    X = df[['Size']]
    y = df['Price']
    predictions = model.predict(X)

    plt.figure()
    plt.scatter(X, y, label="Data Points")
    plt.plot(X, predictions, label="Regression Line", color='red')
    plt.xlabel("Size (sqft)")
    plt.ylabel("Price ($)")
    plt.title("Linear Regression Fit")
    plt.legend()
    plt.show()


# Predict price from user input
def predict_price(model):
    while True:
        user_input = input("\nEnter house size in sqft (or 'q' to quit): ")
        if user_input.lower() == 'q':
            break
        try:
            size = float(user_input)
            if size <= 0:
                raise ValueError("Size must be positive.")
            
            input_df = pd.DataFrame({'Size': [size]})
            prediction = model.predict(input_df)
            print(f"\nEstimated Price: ${prediction[0]:,.2f}")

        except ValueError as e:
            print(f"Invalid input: {e}")

# Main function to run the program
def main():
    file_path = "housing_data_500.csv"  

    df = load_data(file_path)
    if df is None:
        return

    explore_data(df)
    model = train_model(df)
    visualize_model(df, model)
    predict_price(model)


if __name__ == "__main__":
    main()