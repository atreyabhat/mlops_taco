# 1. Use an official lightweight Python base image (Debian-based)
FROM python:3.12-slim

# 2. Set the working directory inside the container
WORKDIR /app

# 3. Copy the requirements file first to leverage Docker's build cache
COPY requirements.txt .

# 4. Install build dependencies, install python packages, and clean up
#    This is the key fix:
#    - 'apt-get update' refreshes package lists
#    - 'apt-get install -y build-essential' installs the C/C++ compilers
#    - 'pip install' runs
#    - 'apt-get purge' removes the compilers to keep the final image small
RUN apt-get update && \
    apt-get install -y build-essential && \
    \
    pip install --no-cache-dir -r requirements.txt && \
    \
    apt-get purge -y build-essential && \
    apt-get autoremove -y && \
    rm -rf /var/lib/apt/lists/*

# 5. Copy your FastAPI application code into the image
COPY ./src/api /app

# 6. Set the environment variable for the MLflow server
# "host.docker.internal" lets the container talk to your laptop's localhost
ENV MLFLOW_TRACKING_URI="http://host.docker.internal:5000"

# 7. Expose the port your API will run on
EXPOSE 8000

# 8. The command to run your application
# This runs your 'main.py' file (which is now at /app/main.py)
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]