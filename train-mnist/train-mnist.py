import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

# Load the data and split it into training and testing sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Normalize the data to values between 0 and 1
x_train, x_test = x_train / 255.0, x_test / 255.0

# Build the neural network model
# model = Sequential([
#     Flatten(input_shape=(28, 28)),  # Flatten the 28x28 images into a 1D array
#     Dense(128, activation='relu'),    # Fully connected layer with 128 units and ReLU activation
#     Dense(10, activation='softmax')   # Output layer with 10 units for 10 digits and softmax activation
# ])

# Build the Convolutional Neural Network model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(10, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
model.fit(x_train, y_train, epochs=5, batch_size=32, validation_data=(x_test, y_test))

# Evaluate the model
test_loss, test_accuracy = model.evaluate(x_test, y_test)
print(f"Test accuracy: {test_accuracy}")

# Show an example result
index = np.random.randint(0, len(x_test))  # Randomly choose an index from the test set
image = x_test[index]
true_label = y_test[index]

# Reshape the image to original size for visualization
image_original = image * 255.0
image_original = image_original.astype(np.uint8)

# Make a prediction
predicted_label = model.predict(np.expand_dims(image, axis=0))
predicted_label = np.argmax(predicted_label)

# Display the image and the prediction
plt.imshow(image_original, cmap='gray')
plt.title(f"True Label: {true_label}, Predicted Label: {predicted_label}")
plt.axis('off')
plt.show()
