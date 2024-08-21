import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error,r2_score
import gradio as gr

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

# Initialize and train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Test the model and check the performance
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
rmse = mse ** 0.5
r2= r2_score(y_test,y_pred)
print(f'Root Mean Squared Error: {rmse}')
print(f'R Squared Value: {r2}')
# Function to predict GPA based on current semester internal marks
def predict_gpa(*internal_marks):
    return model.predict([list(internal_marks)])[0]

# Define the Gradio interface with Number inputs
internal_marks_inputs = [gr.Number(label=f"Subject {i+1} Marks") for i in range(X.shape[1])]

gr.Interface(
    fn=predict_gpa,
    inputs=internal_marks_inputs,
    outputs="number",
    title="GPA Predictor",
    description="Predict GPA based on current semester internal marks."
).launch()
