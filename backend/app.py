
# Import necessary libraries
import joblib
import pandas as pd
from flask import Flask, request, jsonify

# Initialize Flask app
superkart_api = Flask("SuperKart")

# Load the trained model
model = joblib.load(
    "backend_files/superkart_model.joblib"
)

# Home page
@superkart_api.get("/")
def home():
    return "Welcome to the SuperKart System"

# Health-check endpoint
@superkart_api.get("/health")
def health_check():
    return jsonify({
        "status": "ok"
    })

# Single-product prediction endpoint
@superkart_api.post("/v1/predict")
def predict_sales():
    data = request.get_json()

    sample = {
        "Product_Weight": data["Product_Weight"],
        "Product_Sugar_Content": data["Product_Sugar_Content"],
        "Product_Allocated_Area": data["Product_Allocated_Area"],
        "Product_MRP": data["Product_MRP"],
        "Store_Size": data["Store_Size"],
        "Store_Location_City_Type": data["Store_Location_City_Type"],
        "Store_Type": data["Store_Type"],
        "Product_Id_char": data["Product_Id_char"],
        "Store_Age_Years": data["Store_Age_Years"],
        "Product_Type_Category": data["Product_Type_Category"]
    }

    input_data = pd.DataFrame([sample])

    prediction = model.predict(input_data).tolist()[0]

    return jsonify({
        "Sales": prediction
    })

# Batch prediction endpoint
@superkart_api.post("/v1/predictbatch")
def predict_sales_batch():
    uploaded_file = request.files["file"]

    input_data = pd.read_csv(uploaded_file)

    predictions = model.predict(input_data).tolist()

    output_dict = {
        str(index): round(float(prediction), 2)
        for index, prediction in enumerate(predictions)
    }

    return jsonify(output_dict)

# Start the Flask app
if __name__ == "_main_":
    superkart_api.run(
        host="0.0.0.0",
        port=7860,
        debug=False,
        use_reloader=False
    )
