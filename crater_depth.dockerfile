# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set the working directory in the container to /app
WORKDIR /app

# Add the current directory contents into the container at /app
ADD . /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    openjdk-17-jre-headless

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir pyspark pandas matplotlib numpy

# Run the command to start your application
CMD ["python", "./crater_depth_linear_regression.py"]