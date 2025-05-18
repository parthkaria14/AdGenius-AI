# Use the official Python base image
FROM python:3.12

# Set the working directory in the container
WORKDIR /code

# Copy the requirements file into the container
COPY ./requirements.txt /code/requirements.txt

# Install dependencies
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

# Copy the application code into the container
COPY ./app/ /code/app/

# Copy the data folder into the container
COPY ./data/ /code/data/

# Set the command to run the FastAPI application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "80"]