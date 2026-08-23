
import streamlit as st
import pandas as pd
import requests
from io import BytesIO

# ---------------------------------------------------------
# Page configuration
# ---------------------------------------------------------

st.set_page_config(
    page_title="SuperKart Sales Prediction",
    page_icon="📊",
    layout="wide"
)

# Base URL of the Flask backend
BACKEND_URL = "http://backend:7860"

# API request timeout in seconds
REQUEST_TIMEOUT = 30

# ---------------------------------------------------------
# Application title and description
# ---------------------------------------------------------

st.title("SuperKart System")
st.write(
    "Enter the product and store details below to predict the total sales."
)

st.info(
    "Use the individual prediction section for one product, "
    "or upload a CSV file to generate predictions for multiple products."
)

# ---------------------------------------------------------
# Input fields
# ---------------------------------------------------------

st.subheader("Product and Store Details")

col1, col2 = st.columns(2)

with col1:
    Product_Weight = st.number_input(
        "Product Weight",
        min_value=0.0,
        value=12.66,
        step=0.01,
        format="%.2f"
    )

    Product_Sugar_Content = st.selectbox(
        "Product Sugar Content",
        ["Low Sugar", "Regular", "No Sugar"]
    )

    Product_Allocated_Area = st.number_input(
        "Product Allocated Area",
        min_value=0.0,
        value=0.027,
        step=0.001,
        format="%.3f"
    )

    Product_MRP = st.number_input(
        "Product MRP",
        min_value=0.0,
        value=117.08,
        step=0.01,
        format="%.2f"
    )

    Product_Id_char = st.selectbox(
        "Product ID Character",
        ["FD", "DR", "NC"]
    )

with col2:
    Store_Size = st.selectbox(
        "Store Size",
        ["Small", "Medium", "High"]
    )

    Store_Location_City_Type = st.selectbox(
        "Store Location City Type",
        ["Tier 1", "Tier 2", "Tier 3"]
    )

    Store_Type = st.selectbox(
        "Store Type",
        [
            "Supermarket Type1",
            "Supermarket Type2",
            "Supermarket Type3",
            "Departmental Store",
            "Food Mart"
        ]
    )

    Store_Age_Years = st.number_input(
        "Store Age (Years)",
        min_value=0,
        value=16,
        step=1
    )

    Product_Type_Category = st.selectbox(
        "Product Type Category",
        ["Perishables", "Non Perishables"]
    )

# ---------------------------------------------------------
# Create JSON payload
# ---------------------------------------------------------

product_data = {
    "Product_Weight": Product_Weight,
    "Product_Sugar_Content": Product_Sugar_Content,
    "Product_Allocated_Area": Product_Allocated_Area,
    "Product_MRP": Product_MRP,
    "Store_Size": Store_Size,
    "Store_Location_City_Type": Store_Location_City_Type,
    "Store_Type": Store_Type,
    "Product_Id_char": Product_Id_char,
    "Store_Age_Years": Store_Age_Years,
    "Product_Type_Category": Product_Type_Category
}

# ---------------------------------------------------------
# Single prediction
# ---------------------------------------------------------

st.subheader("Single Product Prediction")

if st.button("Predict", type="primary", use_container_width=True):

    try:
        with st.spinner("Generating prediction..."):

            response = requests.post(
                f"{BACKEND_URL}/v1/predict",
                json=product_data,
                timeout=REQUEST_TIMEOUT
            )

        if response.status_code == 200:

            result = response.json()
            predicted_sales = float(result["Sales"])

            st.success("Prediction completed successfully.")

            metric_col1, metric_col2 = st.columns(2)

            with metric_col1:
                st.metric(
                    label="Predicted Product Store Sales Total",
                    value=f"{predicted_sales:,.2f}"
                )

            with metric_col2:
                st.metric(
                    label="Prediction Status",
                    value="Successful"
                )

            st.json(result)

        else:
            st.error(
                f"The prediction API returned an error "
                f"with status code {response.status_code}."
            )

            try:
                st.json(response.json())
            except ValueError:
                st.code(response.text)

    except requests.exceptions.Timeout:
        st.error(
            "The request timed out. Please check whether the backend service "
            "is running and try again."
        )

    except requests.exceptions.ConnectionError:
        st.error(
            "Unable to connect to the prediction API. "
            "Please check the backend URL and confirm that the Flask service is running."
        )

    except requests.exceptions.RequestException as error:
        st.error(f"An error occurred while contacting the API: {error}")

    except KeyError:
        st.error(
            "The API response does not contain the expected 'Sales' field."
        )

    except ValueError:
        st.error(
            "The prediction returned by the API is not a valid numeric value."
        )

# ---------------------------------------------------------
# Batch prediction
# ---------------------------------------------------------

st.divider()
st.subheader("Batch Prediction")

st.write(
    "Upload a CSV file containing the same input features used for "
    "the individual prediction."
)

uploaded_file = st.file_uploader(
    "Upload a CSV file",
    type=["csv"]
)

if uploaded_file is not None:

    try:
        preview_data = pd.read_csv(uploaded_file)

        st.write("Uploaded file preview:")
        st.dataframe(
            preview_data.head(),
            use_container_width=True
        )

        st.caption(
            f"Rows: {preview_data.shape[0]} | "
            f"Columns: {preview_data.shape[1]}"
        )

        # Reset the file pointer before sending the file to the backend
        uploaded_file.seek(0)

        if st.button(
            "Predict for Batch",
            type="primary",
            use_container_width=True
        ):

            with st.spinner("Generating batch predictions..."):

                response = requests.post(
                    f"{BACKEND_URL}/v1/predictbatch",
                    files={
                        "file": (
                            uploaded_file.name,
                            uploaded_file,
                            "text/csv"
                        )
                    },
                    timeout=REQUEST_TIMEOUT
                )

            if response.status_code == 200:

                results = response.json()

                st.success("Predictions completed successfully.")

                try:
                    if isinstance(results, list):
                        results_df = pd.DataFrame(results)

                    elif isinstance(results, dict):

                        if all(
                            not isinstance(value, (list, dict))
                            for value in results.values()
                        ):
                            results_df = pd.DataFrame([results])
                        else:
                            results_df = pd.DataFrame(results)

                    else:
                        results_df = pd.DataFrame(
                            {"Result": [results]}
                        )

                    st.dataframe(
                        results_df,
                        use_container_width=True
                    )

                    # Convert results to CSV for download
                    csv_data = results_df.to_csv(
                        index=False
                    ).encode("utf-8")

                    st.download_button(
                        label="Download Predictions as CSV",
                        data=csv_data,
                        file_name="superkart_batch_predictions.csv",
                        mime="text/csv",
                        use_container_width=True
                    )

                except Exception as error:
                    st.error(
                        f"Unable to display the results as a table: {error}"
                    )
                    st.json(results)

            else:
                st.error(
                    f"The batch prediction API returned an error "
                    f"with status code {response.status_code}."
                )

                try:
                    st.json(response.json())
                except ValueError:
                    st.code(response.text)

    except pd.errors.EmptyDataError:
        st.error("The uploaded CSV file is empty.")

    except pd.errors.ParserError:
        st.error(
            "The uploaded file could not be read as a valid CSV file."
        )

    except requests.exceptions.Timeout:
        st.error(
            "The batch prediction request timed out. "
            "Please check the backend service."
        )

    except requests.exceptions.ConnectionError:
        st.error(
            "Unable to connect to the batch prediction API. "
            "Please check the backend URL and confirm that the Flask service is running."
        )

    except requests.exceptions.RequestException as error:
        st.error(f"An error occurred while contacting the API: {error}")

    except Exception as error:
        st.error(f"Unable to read the uploaded file: {error}")
