import matplotlib.pyplot as plt
import numpy as np
import time
import os
import PIL
import tensorflow as tf
import pathlib
import json
import cv2
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import Dense, Flatten, Input
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.optimizers import Adam

def create_and_train_model():
    # Path to the dataset
    path = "data/flowers"

    # Load the dataset
    data_dir = pathlib.Path(path)

    # Parameters for the loader and the training data is specified
    img_height, img_width = 180, 180
    batch_size = 32
    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="training",
        seed=123,
        label_mode='categorical',
        image_size=(img_height, img_width),
        batch_size=batch_size
    )

    # Parameters for the loader and the validation data is specified
    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="validation",
        seed=123,
        label_mode='categorical',
        image_size=(img_height, img_width),
        batch_size=batch_size
    )

    # The class names are extracted from the dataset
    class_names = train_ds.class_names
    print("Classes:", class_names)

    # specify the input shape size (180 x 180 px)
    input_shape = (180, 180, 3)

    resnet_model = Sequential()
    resnet_model.add(Input(shape=input_shape))
    resnet_model.add(tf.keras.applications.ResNet50(
        include_top=False,
        weights="imagenet",
        input_shape=input_shape,
        pooling='avg',
        classes=5
    ))
    resnet_model.add(Flatten())
    resnet_model.add(Dense(512, activation='relu'))
    resnet_model.add(Dense(5, activation='softmax'))

    # Get summary
    resnet_model.summary()

    # Compile the model
    resnet_model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    # Train the model
    epochs = 10
    history = resnet_model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs
    )

    # Save the trained model
    resnet_model.save('saved_model/flower_classifier.h5')
    
    # Save the training history as a JSON file
    with open('saved_model/training_history.json', 'w') as f:
        # Convert numpy arrays to lists for JSON serialization
        history_dict = {}
        for key, value in history.history.items():
            history_dict[key] = [float(i) for i in value]
        json.dump(history_dict, f)

    # Save class names
    with open('saved_model/class_names.json', 'w') as f:
        json.dump(class_names, f)
    
    return history

def load_and_plot():
    try:
        # Load the training history
        with open('saved_model/training_history.json', 'r') as f:
            history = json.load(f)
        
        # Create a figure with two subplots
        plt.figure(figsize=(12, 5))
        
        # Plot accuracy
        plt.subplot(1, 2, 1)
        plt.plot(history['accuracy'])
        plt.plot(history['val_accuracy'])
        plt.axis(ymin=0.4, ymax=1)
        plt.grid(True)
        plt.title('Model Accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epochs')
        plt.legend(['Train', 'Validation'])

        # Plot loss
        plt.subplot(1, 2, 2)
        plt.plot(history['loss'])
        plt.plot(history['val_loss'])
        plt.grid(True)
        plt.title('Model Loss')
        plt.ylabel('Loss')
        plt.xlabel('Epochs')
        plt.legend(['Train', 'Validation'])

        plt.tight_layout()
        plt.show()

    except FileNotFoundError:
        print("No saved training history found. Please train the model first using 'python model.py train'")

def load_saved_model():
    try:
        model = load_model('saved_model/flower_classifier.h5')
        print("Model loaded successfully!")
        
        # Load class names
        with open('saved_model/class_names.json', 'r') as f:
            class_names = json.load(f)
        print("Available classes:", class_names)
        
        return model, class_names
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        print("Please train the model first using 'python model.py train'")
        return None, None

def test_gpu():
    print("TensorFlow version:", tf.__version__)
    print("CUDA available:", tf.test.is_built_with_cuda())
    print("GPU devices:", tf.config.list_physical_devices('GPU'))

    # GPU speed test
    def run_gpu_test():
        with tf.device('/GPU:0'):
            # Create large tensors
            matrix_size = 5000
            a = tf.random.normal([matrix_size, matrix_size])
            b = tf.random.normal([matrix_size, matrix_size])
            
            start_time = time.time()
            # Matrix multiplication
            c = tf.matmul(a, b)
            # Force evaluation
            _ = c.numpy()
            end_time = time.time()
            
            return end_time - start_time

    print("\nRunning GPU performance test...")
    execution_time = run_gpu_test()
    print(f"Large matrix multiplication took: {execution_time:.2f} seconds")

if __name__ == "__main__":
    import sys
    
    # Create directories if they don't exist
    os.makedirs('saved_model', exist_ok=True)
    
    # Check command line arguments
    if len(sys.argv) > 1:
        if sys.argv[1] == "train":
            print("Training new model...")
            history = create_and_train_model()
            print("\nTraining completed. Plotting results...")
            load_and_plot()
        elif sys.argv[1] == "plot":
            print("Loading and plotting saved training history...")
            load_and_plot()
        elif sys.argv[1] == "test-gpu":
            print("Testing GPU capabilities...")
            test_gpu()
        else:
            print("Invalid argument. Use:")
            print("  'train'    - to train new model")
            print("  'plot'     - to show existing results")
            print("  'test-gpu' - to test GPU capabilities")
    else:
        print("Please specify an argument:")
        print("  'train'    - to train new model")
        print("  'plot'     - to show existing results")
        print("  'test-gpu' - to test GPU capabilities")