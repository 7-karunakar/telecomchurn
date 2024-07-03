from flask import Flask, request, render_template
import joblib
import numpy as np

app = Flask(__name__)

# Load the model, encoders, and column names
model = joblib.load('random_forest_model.pkl')
label_encoders = joblib.load('label_encoders.pkl')
model_columns = joblib.load('model_columns.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        input_data = request.form.to_dict()
    except:
        print("error")

   


    
    
    # Convert numeric inputs to float
    numeric_fields = ['tenure', 'MonthlyCharges', 'TotalCharges']
    for field in numeric_fields:
        input_data[field] = float(input_data[field])
    
    # Encode categorical inputs
    for column, le in label_encoders.items():
        if column in input_data:
            input_data[column] = le.transform([input_data[column]])[0]
    
    # Ensure the input data is in the correct order
    
    input_features = [input_data[column] for column in model_columns]
    input_features = np.array([input_features])

    input_features[0][1] = 1 if input_features[0][1]=="yes" else 0

    
    
    prediction = model.predict(input_features)
    prediction_text = 'Churn' if prediction[0] == 1 else 'No Churn'
    return render_template('result.html', prediction_text=prediction_text)

if __name__ == '__main__':
    app.run(debug=True)
