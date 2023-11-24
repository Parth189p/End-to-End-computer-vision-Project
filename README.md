# Face Recognition Project
[![Project Status: Complete](https://img.shields.io/badge/Project%20Status-Complete-brightgreen)]


Welcome to the Face Recognition Project, an end-to-end machine learning project that combines model building, training, deployment, and MLOps practices. This project showcases my abilities to create scalable, cloud-deployable production-level solutions with various tools and technologies.

## Project Overview

This project implements a face recognition system that recognizes faces of individuals trained on the dataset. The primary components and features include:

- **Model Building:** The face recognition model is built using the TensorFlow framework, employing transfer learning with the VGG16 architecture, specifically VGGFACE. Also used openCV and openCV harcaascade for detecting the face.

- **MLOps Tools:** The project integrates various MLOps tools, such as MLflow for model tracking and monitoring, DVC for data versioning, and Dagshub for collaboration and experiment management.

- **Pipeline Orchestration:** An object-oriented Python pipeline handles various stages, including data ingestion, data preprocessing, model building, model training, model evaluation, and prediction.

- **Data Versioning:** Data versioning and pipeline orchestration are managed using DVC (Data Version Control).

- **CI/CD Pipeline:** GitHub Actions are used to create a Continuous Integration and Continuous Deployment (CI/CD) pipeline, automating the project's build, test, and deployment processes.

- **Docker Deployment:** The project is containerized using Docker for easy deployment and scalability.

- **Cloud Deployment:** The project is deployable on both AWS and Azure, showcasing its cloud compatibility.

- **Flask Web App:** A web application is created using Flask to provide a user-friendly interface for face recognition.

- **TensorBoard Integration:** TensorBoard is used for model visualization and analysis.

- **Custom Log File:** A custom log file structure is implemented to facilitate debugging and error tracking.

## Getting Started

Follow the steps below to get started with the Face Recognition Project:

### Prerequisites

- Python 3.11
- TensorFlow
- MLflow
- DVC
- Flask
- Docker

### Workflow

1. Update config.yaml
2. Update secrets.yaml [Optional]
3. Update params.yaml
4. Update the entity
5. Update the configuration manager in src config
6. Update the components
7. Update the pipeline
8. Update the main.py
9. Update the dvc.yaml

### Initialization

```bash
# Clone this repository
git clone https://github.com/Parth189p/End-to-End-computer-vision-Project.git

# Install project dependencies
pip install -r requirements.txt

# Run the project
python main.py

# If you want to run the Flask app
python app.py

```

## DVC Cmd

1. dvc init
2. dvc repro
3. dvc dag


## Deployment Steps

### AWS CI/CD Deployment with GitHub Action

1. Login to AWS console 
2. Creat I AM user 
3. Creat ECR repo to store
4. Creat EC2 machine
5. Install docker in EC2 machine
6. Configure EC2 as self-hostged runner
7. Setup git-hub secrets

