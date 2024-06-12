import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import numpy as np


# Load the dataset from the specified path
file_path = r'C:\Users\Adeyemo Bolanle\Desktop\BUS Assignments\PREMIER LEAGUE DATASET.csv'
data = pd.read_csv(file_path, encoding='ISO-8859-1')

# Convert 'Cross accuracy %' to numerical by removing '%' and converting to float
data['Cross accuracy %'] = data['Cross accuracy %'].str.replace('%', '').astype(float)

# Features and target variable
X = data[['Goals', 'Goals per match', 'Big chances missed', 'Passes', 'Passes per match', 'Big chances created', 'Crosses', 'Cross accuracy %', 'Through balls', 'Accurate long balls']]
y = data['Assists']

# Normalize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Initialize and train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict on the test set
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Output results
print(f'Mean Squared Error: {mse}')
print(f'R-squared: {r2}')

# Data for Enzo Fernández (not in training set)
enzo_data = {
    'Goals': 1,
    'Goals per match': 0.07,
    'Big chances missed': 6,
    'Passes': 3216,
    'Passes per match': 69.91,
    'Big chances created': 8,
    'Crosses': 59,
    'Cross accuracy %': 17,
    'Through balls': 32,
    'Accurate long balls': 262
}

# Convert Enzo Fernández's data to DataFrame and normalize
enzo_df = pd.DataFrame([enzo_data])
enzo_scaled = scaler.transform(enzo_df)

# Predict assists for Enzo Fernández
predicted_assists_enzo = model.predict(enzo_scaled)[0]

print("\nPredicted Assists for Enzo Fernández:")
print(f"Predicted Assists: {predicted_assists_enzo}")
print(f"Actual Assists: 4")

# Data for Bruno Fernandes (already provided)
fernandes_data = {
    'Goals': 54,
    'Goals per match': 0.34,
    'Big chances missed': 32,
    'Passes': 8555,
    'Passes per match': 53.81,
    'Big chances created': 91,
    'Crosses': 753,
    'Cross accuracy %': 24,
    'Through balls': 107,
    'Accurate long balls': 573
}

# Convert Bruno Fernandes' data to DataFrame and normalize
fernandes_df = pd.DataFrame([fernandes_data])
fernandes_scaled = scaler.transform(fernandes_df)

# Predict assists for Bruno Fernandes
predicted_assists_fernandes = model.predict(fernandes_scaled)[0]

print("\nPredicted Assists for Bruno Fernandes:")
print(f"Predicted Assists: {predicted_assists_fernandes}")
print(f"Actual Assists: 41")

# Predict for Héctor Bellerín using the normalized data
bellerin_data = data[data['Name'] == 'Héctor Bellerín']
bellerin_index = bellerin_data.index[0]  # Index in the original dataset
actual_player_data_bellerin = X.loc[bellerin_index]
actual_assists_bellerin = y.loc[bellerin_index]
bellerin_scaled = scaler.transform([actual_player_data_bellerin])

predicted_assists_bellerin = model.predict(bellerin_scaled)[0]
print("\nHéctor Bellerín's actual player data:")
print(actual_player_data_bellerin)
print(f"Actual Assists: {actual_assists_bellerin}")
print(f"Predicted Assists: {predicted_assists_bellerin}")

# Data for James Maddison
maddison_data = {
    'Goals': 47,
    'Goals per match': 0.25,
    'Big chances missed': 23,
    'Passes': 7211,
    'Passes per match': 37.75,
    'Big chances created': 61,
    'Crosses': 912,
    'Cross accuracy %': 25,
    'Through balls': 88,
    'Accurate long balls': 333
}

# Convert James Maddison's data to DataFrame and normalize
maddison_df = pd.DataFrame([maddison_data])
maddison_scaled = scaler.transform(maddison_df)

# Predict assists for James Maddison
predicted_assists_maddison = model.predict(maddison_scaled)[0]

print("\nPredicted Assists for James Maddison:")
print(f"Predicted Assists: {predicted_assists_maddison}")
print(f"Actual Assists: 41")

# Data for Martin Ødegaard
odegaard_data = {
    'Goals': 31,
    'Goals per match': 0.25,
    'Big chances missed': 11,
    'Passes': 5644,
    'Passes per match': 46.26,
    'Big chances created': 38,
    'Crosses': 263,
    'Cross accuracy %': 29,
    'Through balls': 83,
    'Accurate long balls': 157
}

# Convert Martin Ødegaard's data to DataFrame
odegaard_df = pd.DataFrame([odegaard_data])

# Scale Ødegaard's data using the previously fitted scaler
odegaard_scaled = scaler.transform(odegaard_df)

# Predict assists for Martin Ødegaard
predicted_assists_odegaard = model.predict(odegaard_scaled)[0]

print("\nPredicted Assists for Martin Ødegaard:")
print(f"Predicted Assists: {predicted_assists_odegaard:.2f}")
print(f"Actual Assists: 23")
