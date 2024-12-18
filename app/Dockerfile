FROM tensorflow/tensorflow

# Install dependencies
RUN pip install --no-cache-dir boto3 numpy SQLAlchemy psycopg2-binary fastapi uvicorn pillow python-multipart

# Set the working directory
WORKDIR /app


RUN pip install --no-cache-dir SQLAlchemy psycopg2-binary

COPY *.py .

# Run the FastAPI server
CMD ["uvicorn", "serve:app", "--host", "0.0.0.0", "--port", "8000"]
