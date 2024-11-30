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
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras import metrics
import splitfolders

input_folder = 'data/flowers'
output_folder = 'data/flowers_split'

# Split with a 70%, 15%, 15% ratio for training, testing, and validation respectively (Use seed=42 for reproducibility of split)
splitfolders.ratio(input_folder, output=output_folder, seed=42, ratio=(.7, .15, .15))

# Default Tensorflow to use GPU instead of CPU
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    try:
        # Allow memory growth
        for device in physical_devices:
            tf.config.experimental.set_memory_growth(device, True)
            
        # Set the visible devices
        tf.config.set_visible_devices(physical_devices[0], 'GPU')
    except RuntimeError as e:
        print(e)

def create_and_train_model():
    # Define the base directory where your split dataset is located
    base_dir = 'data/flowers_split'

    # Define the dataset splits
    splits = ['train', 'val', 'test']

    # Define the allowed image extensions
    image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff')

    # Initialize a dictionary to hold counts of each flower class
    counts = {}

    # Iterate over each split
    for split in splits:
        split_dir = os.path.join(base_dir, split)
        counts[split] = {}
        total_images_in_split = 0

        # Get the list of class directories in the current split directory
        classes = [d for d in os.listdir(split_dir) if os.path.isdir(os.path.join(split_dir, d))]

        # Iterate over each class
        for cls in classes:
            class_dir = os.path.join(split_dir, cls)

            # List all files in the class directory
            files = os.listdir(class_dir)

            # Filter out only image files
            images = [f for f in files if f.lower().endswith(image_extensions)]

            # Count the number of images
            num_images = len(images)

            # Store the count in the dictionary
            counts[split][cls] = num_images

            # Keep track of the total number of images in the split
            total_images_in_split += num_images

        # Store the total count for the split
        counts[split]['total'] = total_images_in_split

    # Display the counts
    for split in splits:
        print(f'\n{split.capitalize()} set:')
        print(f'Total images: {counts[split]["total"]}')
        for cls in counts[split]:
            if cls != 'total':
                print(f'  {cls}: {counts[split][cls]} images')

    # Load the dataset
    train_dir = 'data/flowers_split/train'
    val_dir = 'data/flowers_split/val'
    test_dir = 'data/flowers_split/test'

    """ batch_size:

    The batch size in machine learning training is the number of samples processed before the model's internal parameters, like weights, are updated.
    Setting the batch size is essential because it affects the training process, model performance, and computational efficiency. Here’s why it matters:

    Memory Management: Larger batch sizes require more memory, as they load more data into memory at once. Smaller batches are more memory-efficient, 
    making them suitable for systems with limited memory, like a local machine or Raspberry Pi.

    Training Speed: Larger batch sizes tend to make training faster because they enable parallel computation on larger chunks of data. However, they 
    require more powerful hardware (e.g., GPUs), as the computation load is high.

    Model Stability and Convergence: Small batch sizes can introduce more noise into the model’s gradient updates, which may slow down convergence 
    but can help the model find a better generalization (i.e., avoiding local minima). Large batch sizes often make training smoother and more stable, 
    leading to faster convergence, but they can also risk converging to poorer solutions due to reduced noise and generalization ability.

    Epoch and Iteration Balance: With a smaller batch size, the model updates weights more frequently (more iterations per epoch), while larger batch 
    sizes mean fewer updates per epoch. This can influence the number of epochs required for the model to converge effectively.
    """

    # Parameters for the loader and the training data is specified
    img_height, img_width = 224, 224
    batch_size = 32

    """ data_augmentation:

    A data augmentation pipeline with transformations like random horizontal flips, rotations, and zooms helps improve the robustness and generalization 
    of a machine learning model. Here’s how each augmentation type contributes:

    Increasing Data Variety: Data augmentation artificially increases the size and variety of the training dataset without collecting more data. By 
    applying random transformations, the model is exposed to more diverse examples, which helps it generalize better to new, unseen data.

    Reducing Overfitting: When training data is limited or has specific patterns, a model can become too familiar with the training set, leading to 
    overfitting. Random transformations create different versions of each image, encouraging the model to learn broader patterns rather than memorizing 
    specific features.

    Invariance to Transformations:

    Horizontal Flips make the model less sensitive to the left-right orientation of objects, which is useful for tasks where the orientation doesn’t 
    impact the label (e.g., distinguishing cats and dogs).

    Rotations help the model become invariant to slight changes in object orientation, which is crucial in real-world settings where objects aren’t 
    perfectly aligned.

    Zooms expose the model to different scales of objects, helping it learn to recognize objects regardless of their distance from the camera or their size.

    Overall, a data augmentation pipeline enriches the training data, helping the model generalize to real-world variations, improve accuracy, and reduce 
    sensitivity to minor changes in input data.
    """

    # Data augmentation for training
    data_augmentation = tf.keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.2),
        layers.RandomZoom(0.2),
        layers.RandomContrast(0.3),
        layers.RandomBrightness(0.2)
    ])

    """ creating the datasets:

    Dataset Creation:

    Creates two datasets using tf.keras.preprocessing.image_dataset_from_directory:

    Training dataset (80% of data)
    Validation dataset (20% of data)

    Uses categorical labels (one-hot encoded)
    Sets a random seed of 123 for reproducibility
    Stores class names from the training dataset
    """

    # Create raw dataset for training from directory
    raw_train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        train_dir,
        label_mode='categorical',
        image_size=(img_height, img_width),
        batch_size=batch_size
    )

    # Create raw dataset for validation from directory
    raw_val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        val_dir,
        label_mode='categorical',
        image_size=(img_height, img_width),
        batch_size=batch_size
    )

        # Create raw dataset for test from directory
    raw_test_ds = tf.keras.preprocessing.image_dataset_from_directory(
        test_dir,
        label_mode='categorical',
        image_size=(img_height, img_width),
        batch_size=batch_size
    )

    # Get class names
    class_names = raw_train_ds.class_names
    print("Classes:", class_names)

    """
    The AUTOTUNE function in machine learning, specifically in TensorFlow’s tf.data API, is a setting that allows TensorFlow to automatically determine the 
    best configuration for data pipeline operations. AUTOTUNE optimizes the performance of data loading, transforming, and feeding operations by dynamically 
    adjusting the level of parallelism (i.e., the number of CPU threads or resources used) based on available system resources.

    Here’s why AUTOTUNE is valuable:

    Efficient Data Loading: Data loading can become a bottleneck if the model has to wait for data to be loaded or processed before training. By using 
    AUTOTUNE, TensorFlow optimizes data loading to keep the model's training process as smooth and fast as possible.

    Parallel Processing: When you apply transformations (such as data augmentation, shuffling, or batch loading), AUTOTUNE adjusts the number of parallel 
    calls to process data quickly. This is especially useful when working with complex pre-processing, like image augmentation, which can be computationally 
    intensive.

    Dynamic Adjustment: AUTOTUNE can adjust in real-time depending on system load and resource availability. This means it can scale up or down the level of 
    parallel processing to optimize speed without overloading system resources, making it ideal for diverse hardware environments.

    Improved Training Throughput: With AUTOTUNE, training throughput is often maximized, as data is fetched and processed just in time for each training step. 
    This helps prevent "idle time" where the model waits for data, thereby reducing overall training time.

    Dataset Performance Optimization:

    Applies performance optimizations to both datasets:

    .cache() to keep images in memory
    .shuffle(1000) for training data to randomize order
    .prefetch() to optimize data loading
    """

    # Configure datasets for performance
    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = raw_train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
    val_ds = raw_val_ds.cache().prefetch(buffer_size=AUTOTUNE)
    test_ds = raw_test_ds.cache().prefetch(buffer_size=AUTOTUNE)

    """ model creation:

    This model architecture leverages the ResNet50 model as a foundational structure and then customizes it with additional layers for a specific classification 
    task. Here’s a breakdown of each component:

    1. Uses ResNet50 as the Base Model

    Pre-trained on ImageNet: ResNet50 is a deep convolutional neural network pre-trained on the large ImageNet dataset, which contains millions of labeled images 
    across a thousand categories. By using a pre-trained ResNet50, the model benefits from feature representations that capture a wide range of image features.

    Excludes Top Classification Layers: The top classification layers in ResNet50 are specific to ImageNet’s 1,000 classes. By excluding these, we can add custom 
    layers suited to the specific number of classes in the new task.

    Uses Average Pooling: Average pooling is applied to reduce the output dimensions after the convolutional layers, condensing the information and reducing the 
    spatial size while retaining essential features.

    Input Shape of (180, 180, 3): The model expects images resized to 180x180 pixels with 3 color channels (RGB). This shape ensures consistency across all input 
    images.

    2. Initially Freezes All ResNet50 Weights

    Freezing weights means the pre-trained layers of ResNet50 are not updated during the initial training. This step helps preserve the learned features and 
    prevents overfitting, especially if the dataset is small or similar to ImageNet. Once the added layers are trained, these layers can be unfrozen for fine-tuning.

    3. Adds Custom Layers on Top

    Data Augmentation Layer: This layer applies transformations like random rotations, flips, or zooms, helping the model learn from more diverse data and 
    generalize better to new inputs.

    Preprocessing Lambda Layer: A lambda layer may be used for custom preprocessing (e.g., scaling pixel values, standardizing images) to ensure all data is 
    prepared consistently before feeding it to the ResNet50 base.

    ResNet50 Base: The main feature extraction component of the model. This part of the architecture identifies relevant patterns, textures, and structures in the 
    images.

    Dense Layer (256 Units, ReLU Activation): A fully connected layer with 256 neurons and ReLU activation function introduces non-linearity, helping the model 
    learn complex patterns from the features extracted by ResNet50.

    Dropout Layer (50% Dropout): Dropout randomly disables 50% of neurons during training, preventing overfitting by encouraging the model to rely on various 
    features rather than memorizing specific patterns.

    Dense Layer (128 Units, ReLU Activation): Another fully connected layer with 128 neurons, adding more capacity to learn from the features. This layer is also 
    activated by ReLU, which introduces non-linearity.

    Dropout Layer (30% Dropout): This second dropout layer disables 30% of neurons, further reducing overfitting while preserving more neurons than the previous 
    dropout layer.

    Final Classification Layer (Softmax Activation): The final layer classifies the input image. The number of neurons in this layer corresponds to the number of 
    target classes, with softmax activation to output class probabilities for multi-class classification.

    Summary of the Architecture Flow

    Data Input -> Data augmentation -> Preprocessing -> ResNet50 for feature extraction.
    Fully Connected Layers: Dense (256, ReLU) -> Dropout (50%) -> Dense (128, ReLU) -> Dropout (30%).
    Output Layer: Final dense layer with softmax for class probabilities.

    This model structure leverages ResNet50’s powerful feature extraction, enhances generalization with data augmentation and dropout, and customizes classification 
    with dense layers tailored to the specific task.
    """

    # specify the input shape size 224 width, 224 length, 3 channels for RGB
    input_shape = (224, 224, 3)

    # Create the base ResNet50 model initialized with imagenet weights, and without the fully connected layers
    resnet_base = tf.keras.applications.ResNet50(
        include_top=False,
        weights="imagenet",
        input_shape=input_shape,
        pooling='avg'
    )

    # Freeze the pretrained weights in RestNet50 Model
    resnet_base.trainable = False

    # Create the model with our own fully connected layers and set data augmentation
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

    # Label smoothing for categorical crossentropy
    loss = tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1)


    """ model compilation:

    Model compilation is the step where the model is configured with the optimizer, loss function, and evaluation metrics it will use during training. Here’s a 
    breakdown of each component:

    1. Optimizer: Adam with 0.001 Learning Rate

    Adam (Adaptive Moment Estimation): Adam is a popular optimization algorithm for training deep learning models. It combines the advantages of two other optimizers, 
    AdaGrad and RMSProp, to adaptively adjust the learning rate for each parameter based on estimates of the first (mean) and second (variance) moments of the 
    gradients. This helps the model converge faster and more efficiently.

    Learning Rate of 0.001: The learning rate determines the step size the optimizer takes when adjusting the model's weights during training. A learning rate of 
    0.001 is commonly used for Adam and provides a balance between convergence speed and stability. If the rate is too high, the model may fail to converge; if too 
    low, training can be very slow.

    2. Loss Function: Categorical Crossentropy

    Categorical Crossentropy: This loss function is used for multi-class classification tasks where each input belongs to one of multiple classes. It measures the 
    difference between the predicted probability distribution (from the model) and the true probability distribution (the actual labels).

    How it Works: Categorical crossentropy penalizes the model more heavily for being confident in the wrong prediction. If the model assigns a high probability to 
    an incorrect class, the loss will be high. Minimizing this loss encourages the model to improve the accuracy of its class probability predictions.

    3. Metric: Accuracy

    Accuracy: This metric tracks the percentage of correct predictions (i.e., when the predicted class matches the true class) during training and evaluation. 
    Accuracy is an intuitive and straightforward metric for multi-class classification, making it useful for understanding how well the model is performing overall.

    Why Use Accuracy: Since categorical crossentropy can sometimes be harder to interpret, accuracy provides an easily understandable measure of performance, 
    allowing you to track the model’s improvement over time.

    Summary of Model Compilation

    In this compilation setup:

    Adam optimizer with a learning rate of 0.001 adjusts weights efficiently, balancing speed and stability.
    Categorical crossentropy loss helps the model improve its probability predictions for multi-class classification.
    Accuracy metric provides an interpretable evaluation metric for tracking model performance.
    
    Together, these settings make the model effective for multi-class classification, with a focus on both accurate predictions and efficient training.
    """

    # Compile the model
    resnet_model.compile(
        optimizer=Adam(learning_rate=0.00017693),
        loss=loss,
        metrics=['accuracy']
    )

    """ training callbacks:

    Training callbacks are additional functions that help control the training process by responding to specific events during training, like changes in validation 
    loss. In this model, two callbacks are used to improve training efficiency and prevent overfitting:

    1. Early Stopping

    Purpose: Early stopping monitors model performance on the validation set and stops training if the performance no longer improves. This prevents overfitting by 
    stopping training once the model begins to perform worse on the validation set (an indication that it may be "memorizing" the training data instead of 
    generalizing).

    Monitors Validation Loss: The callback watches the validation loss, which reflects how well the model performs on unseen data. If validation loss stops improving, 
    it’s often a sign that further training won’t yield better results.

    Patience of 3 Epochs: "Patience" is the number of epochs to wait for improvement in the monitored metric before stopping training. Here, patience is set to 3 
    epochs, so if the validation loss doesn’t improve for 3 consecutive epochs, training will stop.

    Restores Best Weights: Early stopping saves the model weights from the epoch where validation loss was lowest. If the model overfits in later epochs, these saved 
    weights are restored, ensuring that the best-performing version of the model is saved.

    2. Learning Rate Reduction on Plateau

    Purpose: This callback dynamically adjusts the learning rate to encourage further improvement when the model’s progress slows. Lowering the learning rate allows 
    the model to make finer adjustments to weights, which can be helpful for reaching a better minimum in the loss function.

    Monitors Validation Loss: Like early stopping, this callback also monitors validation loss to detect when the model has reached a "plateau" (i.e., when validation 
    loss stops decreasing).

    Reduces Learning Rate by 80% (factor=0.2): If a plateau is detected, the learning rate is reduced by a factor of 0.2 (or 80%), making it five times smaller. This 
    smaller learning rate helps the model make more precise adjustments, which can sometimes yield better performance.

    Patience of 2 Epochs: Here, patience is set to 2 epochs, so if the validation loss doesn’t improve for 2 consecutive epochs, the learning rate is reduced. This 
    allows the model to explore finer weight adjustments without requiring a restart.

    Summary of Training Callbacks

    Early Stopping: Prevents overfitting by stopping training when validation loss stops improving, with the added benefit of restoring the best weights.
    Learning Rate Reduction: Lowers the learning rate when the model's progress stalls, helping the model continue making improvements by taking smaller steps.

    Together, these callbacks create a more efficient training process that helps the model converge more effectively while reducing the risk of overfitting.
    """

    # Create callbacks for early stopping and learnig rate reduction
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=2
        )
    ]

    """ initial training:

    Initial Training:

    Trains for up to 20 epochs
    Uses the training and validation datasets
    Applies the defined callbacks
    """

    # Train the model
    print("Training the model...")
    epochs = 25
    history = resnet_model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=callbacks
    )

    """ fine-tuning the model:

    Fine-tuning is a technique used to improve model performance by gradually training a pre-trained model on a new dataset. After initial training, 
    fine-tuning allows parts of the pre-trained model’s weights to adjust to better suit the new data. Here’s what each step of this fine-tuning process involves:

    1. Unfreezes the ResNet50 Base Model

    Initially, the ResNet50 model's layers were frozen, meaning they didn’t update during training. This helped retain the knowledge from its original training 
    on ImageNet while allowing the new custom layers to learn.

    Unfreezing means unlocking selected layers in ResNet50, so they can now be updated to better match the new dataset’s specific features.

    2. Keeps the First Many Layers Frozen (All Except Last 30 Layers)

    The layers in a deep model like ResNet50 are hierarchically organized: lower layers detect more general features (like edges and textures), while higher layers 
    detect more specific features (like shapes or object parts).

    By keeping the earlier layers frozen and unfreezing only the last 30 layers, the model preserves general feature knowledge while fine-tuning the higher layers 
    to recognize details relevant to the new task. This approach helps prevent overfitting, as the lower layers are kept stable while only specific patterns are 
    adapted.

    3. Recompiles the Model with a Lower Learning Rate (0.0001)

    Lower Learning Rate (0.0001): When fine-tuning, a lower learning rate is set to avoid making large updates that could disrupt the useful patterns learned in 
    earlier training.

    A smaller learning rate allows for more gradual, precise adjustments, refining the model’s parameters rather than drastically changing them.

    4. Trains for an Additional 10 Epochs

    After unfreezing the selected layers, the model is trained for another 10 epochs. This allows the model to adjust the unfrozen weights to better capture the 
    specific patterns and characteristics of the new dataset.

    Ten epochs provide enough time for the model to fine-tune without risking overfitting or excessive training.

    Summary of Fine-Tuning Process
    Selective Unfreezing: Unlocks the final layers to allow for adjustment, while keeping the earlier layers’ knowledge intact.
    Lower Learning Rate: Ensures adjustments are gradual, helping the model adapt to the new data without disrupting previous knowledge.
    Additional Training: Fine-tunes the model over a limited number of epochs to optimize its performance on the new dataset.

    This approach combines the robustness of the pre-trained ResNet50 model with the flexibility to adapt to new, task-specific features. Fine-tuning with these 
    steps allows the model to achieve higher accuracy on the new task without extensive retraining.
    """

    # Fine-tune the model
    print("\nFine-tuning the model...")

    # Unfreeze ResNet50 base model layers
    resnet_base.trainable = True

    # Freeze the bottom 5 layers
    for layer in resnet_base.layers[:-5]:
        layer.trainable = False

    # Recompile the model
    resnet_model.compile(
        optimizer=Adam(learning_rate=0.000017693),
        loss=loss,
        metrics=['accuracy']
    )

    # Continue training to fine-tune the model
    history_fine = resnet_model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=15,
        callbacks=callbacks
    )

    # Combine histories from initial training and fine-tuning
    for k in history.history:
        history.history[k].extend(history_fine.history[k])

    # Save the trained model
    resnet_model.save('saved_model/flower_classifier')
    
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
        image_resized = cv2.resize(image, (224, 224))
        
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
        # Load class names to get the correct number of classes
        with open('saved_model/class_names.json', 'r') as f:
            class_names = json.load(f)
        num_classes = len(class_names)

        model = tf.keras.models.load_model('saved_model/flower_classifier')
        print("Model weights loaded successfully!")

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