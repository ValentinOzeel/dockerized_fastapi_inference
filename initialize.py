import subprocess
import os
import time

root_path = os.path.abspath(os.path.dirname(__file__))

def end_process(process):
    process.terminate()

def run_command_blocking(command):
    # Wait for completion of the command before going to the next line of code
    subprocess.run(command, shell=True)

def check_image_exists(image_name):
    result = subprocess.run(f"docker inspect --type=image {image_name}", shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return result.returncode == 0

def delete_image(image_name):
    print(f'\n- Deleting existing Docker image "{image_name}"...')
    run_command_blocking(f"docker rmi {image_name}")
    
    
if __name__ == "__main__":
    # Cd to project root (assuming it's where Dockerfile is located at)
    os.chdir(root_path)
    
    image_name = "fastapi-torch-app"
    container_name = "fastapi-torch-container"
    
    # Check if the Docker image already exists (if so delete it)
    if check_image_exists(image_name):
        delete_image(image_name)
        
    # Create fastapi-torch-app Docker image
    print(f'\n- Creating'.upper(), image_name, 'docker image...'.upper())
    run_command_blocking(f"docker build --no-cache -t {image_name} .") 
    #run_command_blocking(f"docker build -t {image_name} .") 
    # Run docker-compose
    print('\n- Launching the container'.upper(), container_name, '  ...')
    run_command_blocking(f"docker run --rm --gpus all -p 8000:8000 --name {container_name} {image_name}")