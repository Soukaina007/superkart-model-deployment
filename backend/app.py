
# Import necessary libraries
import os
from io import BytesIO

import joblib
import pandas as pd
from flask import Flask, jsonify, request

# Initialize Flask app
superkart_api = Flask("SuperKart")
%%writefile backend_files/app.py

import os
from io import BytesIO

import joblib
import pandas as pd
from flask import Flask, jsonify, request

# Initialize Flask app
superkart_api = Flask("SuperKart")

# Exact feature order expected by the trained model
MODEL_FEATURES = [
    "Product_Weight",
    "Product_Sugar_Content",
    "Product_Allocated_Area",
    "Product_MRP",
    "Store_Size",
    "Store_Location_City_Type",
    "Store_Type",
    "Product_Id_char",
    "Store_Age_Years",
    "Product_Type_Category",
]

# Load the trained model
MODEL_PATH = os.path.join(
    os.path.dirname(os.path.abspath(_file_)),
    "superkart_model.joblib"
)

model = joblib.load(MODEL_PATH)

@superkart_api.get("/")
def home():
    return jsonify({
        "message": "Welcome to the SuperKart System"
    })

@superkart_api.post("/v1/predict")
def predict_sales():
    try:
        data = request.get_json(silent=True)

        if not data:
            return jsonify({
                "error": "A JSON request body is required."
            }), 400

        missing_fields = [
            column for column in MODEL_FEATURES
            if column not in data
        ]

        if missing_fields:
            return jsonify({
                "error": "Missing required input fields.",
                "missing_fields": missing_fields
            }), 400

        sample = {
            column: data[column]
            for column in MODEL_FEATURES
        }

        input_data = pd.DataFrame([sample])

        prediction = model.predict(input_data)[0]

        return jsonify({
            "Sales": round(float(prediction), 2)
        })

    except Exception as exc:
        return jsonify({
            "error": f"Single prediction failed: {str(exc)}"
        }), 500

@superkart_api.post("/v1/predictbatch")
def predict_sales_batch():
    try:
        # Check that a file was uploaded
        if "file" not in request.files:
            return jsonify({
                "error": "No CSV file was uploaded. Use the form field name 'file'."
            }), 400

        uploaded_file = request.files["file"]

        if uploaded_file.filename == "":
            return jsonify({
                "error": "The uploaded file has no filename."
            }), 400

        if not uploaded_file.filename.lower().endswith(".csv"):
            return jsonify({
                "error": "Only CSV files are accepted."
            }), 400

        # Read the existing comma-separated CSV
        input_data = pd.read_csv(
            uploaded_file,
            sep=",",
            encoding="utf-8-sig",
            skipinitialspace=True
        )

        # Remove accidental spaces around column names
        input_data.columns = input_data.columns.str.strip()

        # Verify that all model features exist
        missing_columns = [
            column for column in MODEL_FEATURES
            if column not in input_data.columns
        ]

        if missing_columns:
            return jsonify({
                "error": "The CSV is missing required columns.",
                "missing_columns": missing_columns,
                "received_columns": input_data.columns.tolist()
            }), 422

        # Select only the expected model columns in the correct order
        model_input = input_data[MODEL_FEATURES].copy()

        # Generate predictions
        predictions = model.predict(model_input)

        # Return one result per input row
        results = [
            {
                "row": index,
                "Sales": round(float(prediction), 2)
            }
            for index, prediction in enumerate(predictions)
        ]

        return jsonify({
            "predictions": results
        })

    except pd.errors.EmptyDataError:
        return jsonify({
            "error": "The uploaded CSV is empty."
        }), 422

    except pd.errors.ParserError as exc:
        return jsonify({
            "error": f"The CSV could not be parsed: {str(exc)}"
        }), 422

    except Exception as exc:
        return jsonify({
            "error": f"Batch prediction failed: {str(exc)}"
        }), 500

if _name_ == "_main_":
    superkart_api.run(
        host="0.0.0.0",
        port=7860,
        debug=False
    )
