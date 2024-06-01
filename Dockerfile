# Use the official Python image from the Docker Hub
FROM pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime
     
RUN apt-get update && \
    apt-get -y install git && \
    apt-get clean
    
# Set the project root as environment variable
ENV container_workdir /app

# Set the working directory
WORKDIR $container_workdir

# install project requirements
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache -r /tmp/requirements.txt && rm -f /tmp/requirements.txt

# Copy the rest of the application code except what's in .dockerignore
COPY . .

# Go in fastapi_inference
WORKDIR /app/fastapi_inference 

# Expose the port FastAPI will run on
EXPOSE 8000

# Command to run the application
CMD ["uvicorn", "fastapi_serving:app", "--host", "0.0.0.0", "--port", "8000"]