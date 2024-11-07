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
from tensorflow.keras.layers import Dense, Flatten, Input, Dropout
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

    # Data augmentation for training
    data_augmentation = tf.keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.2),
        layers.RandomZoom(0.2),
    ])

    # Create raw datasets first
    raw_train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="training",
        seed=123,
        label_mode='categorical',
        image_size=(img_height, img_width),
        batch_size=batch_size
    )

    raw_val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="validation",
        seed=123,
        label_mode='categorical',
        image_size=(img_height, img_width),
        batch_size=batch_size
    )

    # Get class names before applying pipeline operations
    class_names = raw_train_ds.class_names
    print("Classes:", class_names)

    # Configure datasets for performance
    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = raw_train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
    val_ds = raw_val_ds.cache().prefetch(buffer_size=AUTOTUNE)

    # specify the input shape size (180 x 180 px)
    input_shape = (180, 180, 3)

    # Create the model
    resnet_base = tf.keras.applications.ResNet50(
        include_top=False,
        weights="imagenet",
        input_shape=input_shape,
        pooling='avg'
    )

    # Freeze the pretrained weights
    resnet_base.trainable = False

    # Create the model
    resnet_model = Sequential([
        Input(shape=input_shape),
        data_augmentation,
        layers.Lambda(tf.keras.applications.resnet50.preprocess_input),
        resnet_base,
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(len(class_names), activation='softmax')
    ])

    # Get summary
    resnet_model.summary()

    # Compile the model
    resnet_model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    # Create callbacks
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=3,
            restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=2
        )
    ]

    # Train the model
    print("Training the model...")
    epochs = 20
    history = resnet_model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=callbacks
    )

    # Fine-tune the model
    print("\nFine-tuning the model...")
    resnet_base.trainable = True
    # Freeze the first many layers
    for layer in resnet_base.layers[:-30]:
        layer.trainable = False

    # Recompile the model
    resnet_model.compile(
        optimizer=Adam(learning_rate=0.0001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    # Continue training
    history_fine = resnet_model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=10,
        callbacks=callbacks
    )

    # Combine histories
    for k in history.history:
        history.history[k].extend(history_fine.history[k])

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

def test_prediction(image_path=None):
    """
    Test the model's prediction on a single image.
    If no image_path is provided, it will test on a random image from the dataset.
    """
    try:
        # Load the saved model and class names
        model, class_names = load_saved_model()
        if model is None or class_names is None:
            return
        
        if image_path is None:
            # If no image path provided, use a random image from the dataset
            data_dir = pathlib.Path("data/flowers")
            all_images = list(data_dir.glob('*/*'))
            image_path = str(np.random.choice(all_images))
            print(f"Using random image: {image_path}")
        
        # Read and preprocess the image
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"Error: Could not load image from {image_path}")
            return
            
        # Convert BGR to RGB (cv2 loads in BGR format)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Resize the image to match model's expected sizing
        image_resized = cv2.resize(image, (180, 180))
        
        # Preprocess using ResNet50's preprocessing
        image_array = tf.keras.applications.resnet50.preprocess_input(
            image_resized.astype('float32')
        )
        
        # Expand dimensions
        image_array = np.expand_dims(image_array, axis=0)
        
        # Make prediction
        pred = model.predict(image_array)
        output_class = class_names[np.argmax(pred)]
        confidence = np.max(pred) * 100
        
        # Create figure with three subplots
        plt.figure(figsize=(15, 5))
        
        # Display image
        plt.subplot(1, 3, 1)
        plt.imshow(image_resized)
        plt.title('Input Image')
        plt.axis('off')
        
        # Display prediction results as text
        plt.subplot(1, 3, 2)
        plt.text(0.5, 0.6, f"Predicted Class:\n{output_class}", 
                horizontalalignment='center', fontsize=12)
        plt.text(0.5, 0.4, f"Confidence:\n{confidence:.2f}%",
                horizontalalignment='center', fontsize=12)
        plt.axis('off')
        plt.title('Prediction')
        
        # Display prediction probabilities as bar chart
        plt.subplot(1, 3, 3)
        y_pos = np.arange(len(class_names))
        plt.barh(y_pos, pred[0] * 100)
        plt.yticks(y_pos, class_names)
        plt.xlabel('Probability (%)')
        plt.title('Class Probabilities')
        
        plt.tight_layout()
        plt.show()
        
        # Print results
        print(f"\nPrediction Results:")
        print(f"Predicted Class: {output_class}")
        print(f"Confidence: {confidence:.2f}%")
        print("\nProbabilities for each class:")
        for class_name, probability in zip(class_names, pred[0]):
            print(f"{class_name}: {probability*100:.2f}%")
            
    except Exception as e:
        print(f"Error during prediction: {str(e)}")

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
        elif sys.argv[1] == "predict":
            if len(sys.argv) > 2:
                # If image path is provided
                image_path = sys.argv[2]
                print(f"Testing prediction on image: {image_path}")
                test_prediction(image_path)
            else:
                # If no image path provided, use random image
                print("Testing prediction on random image...")
                test_prediction()
        else:
            print("Invalid argument. Use:")
            print("  'train'              - to train new model")
            print("  'plot'               - to show existing results")
            print("  'test-gpu'           - to test GPU capabilities")
            print("  'predict [filepath]' - to test prediction on an image")
    else:
        print("Please specify an argument:")
        print("  'train'              - to train new model")
        print("  'plot'               - to show existing results")
        print("  'test-gpu'           - to test GPU capabilities")
        print("  'predict [filepath]' - to test prediction on an image")