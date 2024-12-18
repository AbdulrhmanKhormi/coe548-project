import os
import numpy as np
import tensorflow as tf
import boto3
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import ModelCheckpoint, Callback
from sqlalchemy import create_engine, Column, Integer, Float, MetaData, Table
from sqlalchemy.orm import sessionmaker

# Get environment variables
BUCKET_NAME = os.getenv("BUCKET_NAME", "train")
DB_USER = os.getenv("DB_USER", "train")
DB_PASSWORD = os.getenv("DB_PASSWORD", "train123")
DB_HOST = os.getenv("DB_HOST", "postgres")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_NAME = os.getenv("DB_NAME", "train_db")

# S3 Client setup
s3 = boto3.client("s3")

# Database setup
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


# Callback to log results to PostgreSQL
class PostgresLogger(Callback):
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        new_record = {
            "epoch": epoch + 1,
            "accuracy": logs.get("accuracy", 0.0),
            "val_accuracy": logs.get("val_accuracy", 0.0),
            "loss": logs.get("loss", 0.0),
            "val_loss": logs.get("val_loss", 0.0),
        }

        # Insert statement
        insert_stmt = training_results_table.insert().values(new_record)

        # Create session to commit the transaction
        with Session() as session:
            session.execute(insert_stmt)
            session.commit()


# Callback to save the model after each epoch
class SaveModelAfterCheckpoint(Callback):
    def on_epoch_end(self, epoch, logs=None):
        print(f"saving the model after {epoch}")
        model.save("/shared/mnist_model.keras")


# Function to load the dataset from S3
def load_data_from_s3(s3_bucket, x_train_key, y_train_key, x_test_key, y_test_key):
    local_path = "/tmp/mnist/"
    os.makedirs(local_path, exist_ok=True)

    # Download files from S3
    s3.download_file(s3_bucket, x_train_key, os.path.join(local_path, "x_train.npy"))
    s3.download_file(s3_bucket, y_train_key, os.path.join(local_path, "y_train.npy"))
    s3.download_file(s3_bucket, x_test_key, os.path.join(local_path, "x_test.npy"))
    s3.download_file(s3_bucket, y_test_key, os.path.join(local_path, "y_test.npy"))

    # Load the data into numpy arrays
    x_train = np.load(os.path.join(local_path, "x_train.npy"))
    y_train = np.load(os.path.join(local_path, "y_train.npy"))
    x_test = np.load(os.path.join(local_path, "x_test.npy"))
    y_test = np.load(os.path.join(local_path, "y_test.npy"))

    return (x_train, y_train), (x_test, y_test)


# S3 Dataset keys
x_train_key = "mnist/x_train.npy"
y_train_key = "mnist/y_train.npy"
x_test_key = "mnist/x_test.npy"
y_test_key = "mnist/y_test.npy"

# Load the MNIST dataset from S3
(x_train, y_train), (x_test, y_test) = load_data_from_s3(
    BUCKET_NAME, x_train_key, y_train_key, x_test_key, y_test_key
)

# Normalize the pixel values to between 0 and 1
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

# Build the model
model = models.Sequential(
    [
        layers.Flatten(input_shape=(28, 28)),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.2),
        layers.Dense(10, activation="softmax"),
    ]
)

# Compile the model
model.compile(
    optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
)

# Setup model checkpoint callback
checkpoint_callback = ModelCheckpoint(
    "model_epoch_{epoch:02d}.keras",
    save_best_only=False,
    save_weights_only=False,
    verbose=1,
)

# Train the model
history = model.fit(
    x_train,
    y_train,
    epochs=100,
    batch_size=64,
    validation_data=(x_test, y_test),
    callbacks=[checkpoint_callback, PostgresLogger(), SaveModelAfterCheckpoint()],
)

# Save the trained model for inference
model.save("/shared/mnist_model.keras")
print("final model saved")

# Evaluate the model on test data
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
print(f"Test accuracy: {test_acc:.4f}")
