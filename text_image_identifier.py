import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt

# Load the MNIST dataset
mnist = tf.keras.datasets.mnist

# Split dataset into training and testing sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Normalize the data (scaling to [0, 1])
x_train, x_test = x_train / 255.0, x_test / 255.0

# Build a simple neural network model
model = models.Sequential([
    layers.Flatten(input_shape=(28, 28)),  # Flatten 28x28 images to a 1D vector
    layers.Dense(128, activation='relu'),  # First fully connected layer with 128 units
    layers.Dropout(0.2),  # Add dropout for regularization
    layers.Dense(10, activation='softmax')  # Output layer for 10 digits (0-9)
])

# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
model.fit(x_train, y_train, epochs=5)

# Evaluate the model on test data
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)

print('\nTest accuracy:', test_acc)

# Make predictions
predictions = model.predict(x_test)

# Display the first test image and prediction
plt.imshow(x_test[59], cmap=plt.cm.binary)
plt.title(f"Predicted: {predictions[0].argmax()}, Actual: {y_test[0]}")
plt.show()

