import joblib
import pandas as pd


#Load the Model
model = joblib.load('House_Price_Prediction_Model.joblib')

#get the user input
def get_user_input():
    # Get input from the user
    county = input("Enter county: ")
    living_space = float(input("Enter living space: "))
    latitude = float(input("Enter latitude: "))
    zip_code_density = float(input("Enter zip code density: "))
    beds = int(input("Enter number of beds: "))
    longitude = float(input("Enter longitude: "))
    baths = int(input("Enter number of baths: "))
    zip_code_population = int(input("Enter zip code population: "))
    median_household_income = float(input("Enter median household income: "))

    # Create a dictionary with the input data
    input_data = {
        "County": [county],
        "Living Space": [living_space],
        "Latitude": [latitude],
        "Zip Code Density": [zip_code_density],
        "Beds": [beds],
        "Longitude": [longitude],
        "Baths": [baths],
        "Zip Code Population": [zip_code_population],
        "Median Household Income": [median_household_income]
    }

    return input_data

#Predicting the house price based on the user input
def predict_price(input_data):
    input_df = pd.DataFrame(input_data)
    #make prediction
    predicted_price = model.predict(input_df)[0]
    return predicted_price

def main():
    print("Welcome to the House Price Prediction App!")

    while True:
        input_data = get_user_input()
        predicted_price = predict_price(input_data)
        print(f"The predicted price of the house is: ${predicted_price:.2f}")

        choice = input("Do you want to make another prediction? (yes/no): ")
        if choice.lower() != 'yes':
            break

    print("Thank you for using the House Price Prediction App!")

if __name__ == "__main__":
    main()