# Clouds Classification Project

This project is about acquisition, cleansing, and analysis of data pertaining to cloud formation. Its primary objective is to leverage machine learning methodologies to construct a model capable of accurately predicting cloud formation patterns.

The project encompasses several key phases: sourcing data from an online repository, meticulously cleaning and structuring the dataset, engineering pertinent features for model training, partitioning the data into distinct training and testing subsets, conducting model training, and meticulously assessing the model's efficacy on the test subset. Each stage's output is meticulously archived within a local directory to facilitate traceability, while all associated artifacts are promptly uploaded to an S3 bucket for seamless access and effortless sharing.

## Pipeline Overview
The pipeline consists of the following steps:

1. **Acquire Data:** Downloads the clouds data from an online repository and saves it to disk.
2. **Create Dataset:** Creates a structured dataset from the raw data and saves it to disk.
3. **Generate Features:** Enriches the dataset with features for model training and saves it to disk.
4. **EDA:** Generates statistics and visualizations for summarizing the data and saves them to disk.
5. **Train Model:** Splits the data into train/test set and trains a random forest model. The trained model is saved to disk along with the train/test datasets.
6. **Score Model:** Scores the trained model on the test dataset and saves the scores to disk.
7. **Evaluate Performance:** Evaluates the performance of the trained model based on various metrics such as accuracy, AUC, etc. The evaluation results are saved to disk.
8. **Upload Artifacts:** Uploads all the artifacts generated during the pipeline execution to an S3 bucket.

## Setting Up the Project

### Clone the Repository
Clone this repository to your local machine using Git:

```
$ git clone https://github.com/MSIA/uaq7345
```

### Install Requirements
Navigate to the cloned repository and install the required packages using `pip`:

```
$ cd uaq7345
$ pip install -r requirements.txt
```

### Build AWS S3 Bucket

Build your AWS S3 Bucket for uploading the artifacts. Once your bucket is created, change the bucket_name in `config/default-config.yaml` to the name of your bucket.


### AWS Configuration

The `aws configure` command is used to set up AWS credentials on your local machine. To run the `aws configure` command, follow these steps:

1. Open a terminal or command prompt.
2. Install the AWS CLI by running the following command: `pip install awscli`.
3. Run the `aws configure` command.
4. Enter your access key, secret key, default region, and default output format.
5. Press Enter to leave any of the fields blank.
6. Verify that the configuration file was created by running the following command: `cat ~/.aws/config` on macOS and Linux or `type C:\Users\USERNAME\.aws\config` on Windows.

Note: Make sure to keep your access key and secret key secure and never share them with anyone.

Then, you need to configured an aws profile, do so with the following:

```
aws configure sso --profile personal-sso-admin
```

You may name the profile whatever you like so long as it helps you identify the Account/Role being used. In this guide, we will use personal-sso-admin.

If you have already done this, you may need to login to refresh credentials. Once you have done so, verify your identity using sts.

```
aws sso login --profile personal-sso-admin
aws sts get-caller-identity --profile personal-sso-admin
```

#### Using a named profile from credentials file (recommended)

The cleanest way to authenticate your boto3 calls is to used the shared credentials file which can be used by any number of applications or libraries. This file has sections for each named profile so you simply need to tell you application which profile to use. Setting the profile can (and should) be done with environment variables as seen below:

```
export AWS_PROFILE=personal-sso-admin
```

#### Setting credential environment variables directly

If you need to have your credentials stored in environment variables, you can do so via the following.

```
eval $(aws configure export-credentials --format env --profile personal-sso-admin)
```

### Fetch Data

This project includes a Python script called `acquire_data()` that can be used to acquire data from a specified URL and save it to a local file path.

To use the `acquire_data()` script, follow these steps:

1. Open a terminal window and navigate to the project's `src` directory.
2. Open the Python interpreter by typing `python` in the terminal and pressing Enter.
3. Import the `acquire_data()` function by typing `from acquire_data import acquire_data` in the Python interpreter and pressing Enter.
4. Call the `acquire_data()` function with two arguments: the URL from which to acquire the data, and the local file path to which to save the acquired data. For example: `acquire_data('https://example.com/data.csv', '/path/to/local/file.csv')`.
5. The script will download the data from the specified URL and save it to the specified local file path. If the download and save are successful, the script will print a success message to the terminal.


### Execute the Pipeline

To execute the pipeline, run the following command:

```
$ python pipeline.py --config config/default-config.yaml
```

This command will use the configuration specified in `config/default-config.yaml` to execute the pipeline. If you want to use a different configuration file, specify the path to the file using the `--config` argument:

```
$ python pipeline.py --config path/to/custom-config.yaml
```

### Run the Pytest

To execute the pytest on the generate_features.py script, run the following command:

```
$ pytest tests/test_generate_features.py
```

### Upload Artifacts
If you want to upload the artifacts generated during the pipeline execution to an S3 bucket, make sure to configure the S3 credentials in `config/default-config.yaml`. Then, run the following command:

```
$ python src/aws_utils.py --config config/default-config.yaml --artifacts artifacts/
```

Note that you need to replace `artifacts/` with the path to the directory containing the artifacts you want to upload.


## Using Docker

Running your application in a Docker container can provide a challenge since the container will not share the files nor environment variables with your local client

### Build the Docker image

```bash
docker build -t hw2-pipeline -f dockerfiles/DockerfilePPL .
```

### Run the entire model pipeline

```bash
docker run -v ~/.aws:/root/.aws -e AWS_PROFILE=personal-sso-admin hw2-pipeline
```

### Build the Docker image for tests

```bash
docker build -t hw2-test -f dockerfiles/DockerfileTest .
```

### Run the tests

```bash
docker run -v ~/.aws:/root/.aws -e AWS_PROFILE=personal-sso-admin hw2-test
```


