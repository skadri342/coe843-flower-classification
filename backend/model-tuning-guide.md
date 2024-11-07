# Model Parameter Tuning Guide for Improved Accuracy with Image Classification

## 1. Data Augmentation Parameters
```python
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.2),
    layers.RandomZoom(0.2),
])
```

### Possible Modifications:
- **Add More Augmentation Types:**
  - `layers.RandomBrightness(factor=0.2)`
  - `layers.RandomContrast(factor=0.2)`
  - `layers.RandomTranslation(height_factor=0.1, width_factor=0.1)`
  - `layers.RandomCrop(height, width)`
  
- **Adjust Intensity:**
  - Increase/decrease rotation angle (0.1 to 0.4)
  - Modify zoom range (0.1 to 0.3)
  - Tune brightness and contrast factors

**Benefits:** Helps prevent overfitting and improves model generalization

## 2. Dataset Loading Parameters
```python
img_height, img_width = 180, 180
batch_size = 32
```

### Tunable Parameters:
- **Image Dimensions:**
  - 224×224 (standard ResNet size)
  - 299×299 (Inception-v3 size)
  - 380×380 (higher resolution)

- **Batch Size Options:**
  - Small (16-32): Better generalization, more noise in training
  - Medium (64-128): Balance between speed and generalization
  - Large (256-512): Faster training, might require more epochs

**Note:** Larger images require more memory and computation time

## 3. Model Architecture Parameters
```python
resnet_model = Sequential([
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.3),
])
```

### Adjustable Components:
- **Dense Layers:**
  - Number of layers (2-5)
  - Units per layer (128, 256, 512, 1024)
  
- **Dropout Rates:**
  - Light (0.2-0.3): Less regularization
  - Medium (0.4-0.5): Standard regularization
  - Heavy (0.6-0.7): Strong regularization

- **Activation Functions:**
  - ReLU (standard)
  - LeakyReLU (alpha=0.01)
  - SELU (self-normalizing)
  - ELU (alpha=1.0)

- **Additional Layers:**
```python
layers.BatchNormalization()
layers.GlobalAveragePooling2D()
layers.GlobalMaxPooling2D()
```

## 4. Base Model Selection
```python
resnet_base = tf.keras.applications.ResNet50(
    include_top=False,
    weights="imagenet",
    input_shape=input_shape,
    pooling='avg'
)
```

### Alternative Models:
- **EfficientNet Family:**
  - EfficientNetB0 (smallest)
  - EfficientNetB3 (medium)
  - EfficientNetB7 (largest)

- **ResNet Family:**
  - ResNet50V2
  - ResNet101
  - ResNet152

- **Other Architectures:**
  - DenseNet121
  - InceptionV3
  - MobileNetV3

## 5. Training Parameters
```python
model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)
```

### Optimization Options:
- **Learning Rates:**
  - High (1e-3): Faster learning, might be unstable
  - Medium (1e-4): Good balance
  - Low (1e-5): Slower but more stable

- **Optimizers:**
  - Adam (standard)
  - RMSprop (good for RNNs)
  - SGD with momentum (0.9)
  - AdamW (Adam with weight decay)

- **Learning Rate Schedules:**
```python
tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=1e-3,
    decay_steps=1000,
    decay_rate=0.9
)
```

## 6. Fine-tuning Parameters
```python
for layer in resnet_base.layers[:-30]:
    layer.trainable = False
```

### Tunable Aspects:
- **Unfrozen Layers:**
  - Last 10 layers (minimal fine-tuning)
  - Last 30 layers (moderate fine-tuning)
  - Last 50+ layers (extensive fine-tuning)

- **Fine-tuning Learning Rate:**
  - Usually 10% of initial rate
  - Try: 1e-5 to 1e-6

- **Fine-tuning Duration:**
  - Short (5-10 epochs)
  - Medium (10-20 epochs)
  - Long (20+ epochs)

## 7. Callback Parameters
```python
callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=3
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2,
        patience=2
    )
]
```

### Adjustable Settings:
- **Early Stopping:**
  - Patience: 3-10 epochs
  - Min delta: 1e-4 to 1e-2
  - Monitor: 'val_loss' or 'val_accuracy'

- **Learning Rate Reduction:**
  - Reduction factor: 0.1 to 0.5
  - Patience: 2-5 epochs
  - Min delta: 1e-4 to 1e-2

- **Additional Callbacks:**
```python
tf.keras.callbacks.ModelCheckpoint(
    filepath='best_model.h5',
    save_best_only=True,
    monitor='val_accuracy'
)
```

## Best Practices for Experimentation

1. **Systematic Approach:**
   - Change one parameter at a time
   - Document all changes and results
   - Use version control for model iterations

2. **Monitoring:**
   - Track both training and validation metrics
   - Watch for overfitting signs
   - Monitor resource usage (memory, GPU)

3. **Validation Strategy:**
   - Use cross-validation for robust evaluation
   - Keep a separate test set
   - Consider real-world data distribution

4. **Resource Management:**
   - Start with smaller models/datasets
   - Gradually increase complexity
   - Use GPU profiling tools

## Common Issues and Solutions

1. **Overfitting:**
   - Increase data augmentation
   - Add dropout layers
   - Reduce model capacity
   - Implement early stopping

2. **Underfitting:**
   - Increase model capacity
   - Reduce regularization
   - Train for more epochs
   - Increase learning rate

3. **Unstable Training:**
   - Reduce learning rate
   - Increase batch size
   - Add batch normalization
   - Check for data issues

Always benchmark changes against a baseline model and maintain good documentation of experiments.
