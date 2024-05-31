import os
from pathlib import Path
import logging
import logging.config
import boto3
import joblib
import yaml
import streamlit as st
from botocore.exceptions import NoCredentialsError
import src.present_interface as pi

logging.config.fileConfig("config/logging/local.conf")
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("clouds")

# Load configuration reference
CONFIG_REF = os.getenv("CONFIG_REF", "config/config.yaml")

def main() -> None:
    """
    Main function to run the Streamlit app.
    This function loads the configuration, sets up the output directory, 
    loads the preprocessor and model, and presents the user interface.
    """
    s3_ = boto3.resource("s3")
    bucket_name = "423-hw3-zaf6599"
    object_name = "config.yaml"
    file_name = "config.yaml"

    # Try to download the configuration file from S3 bucket
    try:
        s3_.Bucket(bucket_name).download_file(object_name, file_name)
        logging.info("Download Successful")
    except NoCredentialsError:
        logging.error("No AWS credentials were found")

    # Load the configuration file
    with open(file_name, "r") as ymlfile:
        config = yaml.safe_load(ymlfile)
    bucket_name = config["aws"]["bucket_model_artifacts"]
    artifacts = Path("artifacts")
    artifacts.mkdir(parents=True, exist_ok=True)

    # Create a cache function to load model from S3
    @st.cache_resource
    def load_model_from_s3(bucket_name: str, object_name: str, local_file_name: str):
        """
        Download a file from S3 and load a model.
        Args:
            bucket_name (str): Name of the S3 bucket.
            object_name (str): Path in S3.
            local_file_name (str): Path where the file will be saved locally.
        Returns:
            The loaded model.
        """
        # Create an S3 client
        s3_ = boto3.client("s3")

        # Try to download the model from S3
        # Try to download the model from S3
        try:
            s3_.download_file(bucket_name, object_name, local_file_name)
            logging.info("Model downloaded successfully from S3")
        except NoCredentialsError:
            logging.error("No AWS credentials were found")
            return None
        except FileNotFoundError as e:
            logging.error("File not found: %s", e)
            return None

        # If the model was downloaded successfully, try to load it
        try:
            model = joblib.load(local_file_name)
            logging.info("Model loaded successfully")
        except FileNotFoundError as e:
            logging.error("File not found: %s", e)
            return None

        return model

    # Define Streamlit title and sidebar header
    st.title("Cloud Prediction")
    st.sidebar.header("User Input Parameters")

    # Sidebar to choose model
    model_choice = st.sidebar.selectbox(
        "Choose the model",
        ("Random Forest", "Logistic Regression")
    )
    local_model_file_name = "local_model.pkl"
    # Depending on the choice, instantiate the correct model
    if model_choice == "Random Forest":
        model_s3_key = config["aws"]["random_forest_model_name"]
    elif model_choice == "Logistic Regression":
        model_s3_key = config["aws"]["logistic_regression_model_name"]
    model = load_model_from_s3(bucket_name, model_s3_key, local_model_file_name)

    # Present user interface
    logger.info("Presenting user interface...")
    pi.present_interface(model, config["present_interface"], config["generate_features"], config["prediction"])

if __name__ == "__main__":
    main()
