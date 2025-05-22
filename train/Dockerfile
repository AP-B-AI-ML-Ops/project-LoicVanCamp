FROM python:3.11-slim

WORKDIR /app

# System dependencies (for pandas, scikit-learn, enz.)
RUN apt-get update && apt-get install -y gcc

# Python dependencies
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# Set environment variables
CMD ["python", "scripts/main.py"]