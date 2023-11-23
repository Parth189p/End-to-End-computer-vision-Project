# Use the official Python image as a parent image
FROM python:3.9-slim-buster

# Install system dependencies
RUN apt-get update -y && \
    apt-get install -y libgl1-mesa-glx libglib2.0-0 libsm6 libxext6 libxrender-dev

# Install AWS CLI
RUN apt-get install awscli -y


# Set the working directory to /app
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install Python dependencies
RUN pip install -r requirements.txt

# Expose port 8080
EXPOSE 8080

# Set the command to run your application
CMD ["python3", "app.py"]
