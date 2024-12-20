FROM python:3.13-slim

# Install required Python packages
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copy your script into the container
COPY your_script.py /app/SMP_Data_Processing.py

# Set the working directory
WORKDIR /app

# Command to run the script
CMD ["python", "SMP_Data_Processing.py"]
