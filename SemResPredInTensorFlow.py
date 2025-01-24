import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf
from sklearn.metrics import mean_squared_error, r2_score
import gradio as gr
import numpy as np

# Load the dataset
file_path = 'sem.csv'
data = pd.read_csv(file_path)

# Drop the 'S.NO' column as it's not needed for prediction
data = data.drop(columns=['S.NO'])

# Separate features (internal marks) and target (GPA)
X = data.drop(columns=['GPA'])
y = data['GPA']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build the TensorFlow model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(X.shape[1],)),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(1)
])

# Compile the model
model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# Train the model
history = model.fit(X_train, y_train, 
                   epochs=100, 
                   batch_size=32,
                   validation_split=0.2,
                   verbose=1)

# Test the model and check the performance
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
rmse = mse ** 0.5
r2 = r2_score(y_test, y_pred)
print(f'Root Mean Squared Error: {rmse}')
print(f'R Squared Value: {r2}')

# Function to predict GPA based on current semester internal marks
def predict_gpa(*internal_marks):
    input_data = np.array(list(internal_marks)).reshape(1, -1)
    return float(model.predict(input_data)[0][0])

# Define the Gradio interface with Number inputs
internal_marks_inputs = [gr.Number(label=f"Subject {i+1} Marks") for i in range(X.shape[1])]

gr.Interface(
    fn=predict_gpa,
    inputs=internal_marks_inputs,
    outputs="number",
    title="GPA Predictor",
    description="Predict GPA based on current semester internal marks."
).launch(share=True)
