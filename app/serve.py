from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from tensorflow.keras.models import load_model
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import create_engine, Column, Integer, Float, MetaData, Table
from sqlalchemy.orm import sessionmaker
from PIL import Image
import numpy as np
import io
import uvicorn
import os

# Initialize FastAPI app
app = FastAPI()

# Add CORS middleware
origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  # List of origins you want to allow
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods
    allow_headers=["*"],  # Allow all headers
)

# Database setup
DB_USER = os.getenv("DB_USER", "train")
DB_PASSWORD = os.getenv("DB_PASSWORD", "train123")
DB_HOST = os.getenv("DB_HOST", "postgres")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_NAME = os.getenv("DB_NAME", "train_db")
DATABASE_URL = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
engine = create_engine(DATABASE_URL)
Session = sessionmaker(bind=engine)
session = Session()

# Define metadata and results table
metadata = MetaData()

training_results_table = Table(
    "training_results",
    metadata,
    Column("id", Integer, primary_key=True),
    Column("epoch", Integer, nullable=False),
    Column("accuracy", Float, nullable=False),
    Column("val_accuracy", Float, nullable=False),
    Column("loss", Float, nullable=False),
    Column("val_loss", Float, nullable=False),
)

# Create table if it doesn't exist
metadata.create_all(engine)


# Define training results table
training_results_table = Table("training_results", metadata, autoload_with=engine)

# Global model variable
model = None


@app.get("/", response_class=HTMLResponse)
async def upload_page():
    return HTMLResponse(
        content="""<!DOCTYPE html>
<html lang=\"en\">
<head>
    <meta charset=\"UTF-8\">
    <meta name=\"viewport\" content=\"width=device-width, initial-scale=1.0\">
    <title>File Upload</title>
    <link rel=\"stylesheet\" href=\"/static/style.css\">
    <script>
        async function handleFormSubmit(event) {
            event.preventDefault();
            const formData = new FormData(event.target);
            const response = await fetch('/predict/', {
                method: 'POST',
                body: formData
            });
            const resultDiv = document.getElementById('result');
            if (response.ok) {
                const result = await response.json();
                resultDiv.textContent = `Predicted Class: ${result.predicted_class}`;
            } else {
                const error = await response.json();
                resultDiv.textContent = `Error: ${error.error}`;
            }
        }

        async function loadModel() {
            const response = await fetch('/load-model/', {
                method: 'POST'
            });
            const resultDiv = document.getElementById('result');
            if (response.ok) {
                resultDiv.textContent = "Model loaded successfully.";
            } else {
                const error = await response.json();
                resultDiv.textContent = `Error: ${error.error}`;
            }
        }

        async function showResults() {
            const response = await fetch('/show-results/', {
                method: 'GET'
            });
            const resultDiv = document.getElementById('result');
            if (response.ok) {
                const results = await response.json();
                resultDiv.textContent = `Results: ${JSON.stringify(results)}`;
            } else {
                const error = await response.json();
                resultDiv.textContent = `Error: ${error.error}`;
            }
        }
    </script>
</head>
<body>
    <h1>Upload an Image</h1>
    <button onclick=\"loadModel()\">Load Model</button>
    <button onclick=\"showResults()\">Show Results</button>
    <form onsubmit=\"handleFormSubmit(event)\" enctype=\"multipart/form-data\">
        <input type=\"file\" name=\"file\" accept=\"image/*\" required>
        <button type=\"submit\">Get Prediction</button>
    </form>
    <div id=\"result\" style=\"margin-top: 20px; font-size: 18px; color: blue;\"></div>
</body>
</html>"""
    )


@app.post("/load-model/", response_class=JSONResponse)
async def load_model_endpoint():
    global model
    try:
        model = load_model("/shared/mnist_model.keras")
        return {"message": "Model loaded successfully."}
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)


@app.post("/predict/", response_class=JSONResponse)
async def predict(file: UploadFile = File(...)):
    global model
    if model is None:
        return JSONResponse(content={"error": "Model not loaded."}, status_code=400)

    try:
        # Read the uploaded image
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data)).convert("L")
        image = image.resize((28, 28))
        image = np.array(image) / 255.0
        image = np.reshape(image, (1, 28, 28))

        # Perform prediction
        prediction = model.predict(image)
        predicted_class = np.argmax(prediction, axis=1)[0]

        return {"predicted_class": int(predicted_class)}
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)


@app.get("/show-results/", response_class=JSONResponse)
async def show_results():
    try:
        with engine.connect() as conn:
            query = training_results_table.select()
            results = conn.execute(query).fetchall()

            # Get column names from the table
            columns = training_results_table.columns.keys()
            print(columns)

            # Convert each tuple result to a dictionary using column names
            formatted_results = [dict(zip(columns, result)) for result in results]

        return formatted_results
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)


# Run the app if this is the main module
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
