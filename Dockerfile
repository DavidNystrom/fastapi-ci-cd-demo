# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set working directory in the container
WORKDIR /app

# Install dependencies (copy requirements and pip install)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code
COPY . .

# Expose port 8000 ( FastAPI will run on this port )
EXPOSE 8000

# Run the app with Uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]