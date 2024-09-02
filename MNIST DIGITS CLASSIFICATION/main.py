import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
import numpy as np
import matplotlib.pyplot as plt



#Load MNIST dataset
(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()

#Normalize the images to be in the range [0,1]
train_images = train_images/255.0
test_images = test_images/255.0

#Flatten the images from 28*28 to 784 dimension [1d array]
train_images= train_images.reshape((train_images.shape[0] ,28*28))
test_images = test_images.reshape((test_images.shape[0], 28*28))

# ------------------------------BUILD THE NEURAL NETWORK----------------------------------------

#Define the model architecture
model = models.Sequential([
    layers.Dense(128, activation = 'relu', input_shape = (784,), kernel_regularizer = regularizers.l2(0.001)), #Hidden layer with 128 neurons
    layers.Dropout(0.5),  # 50% dropout rate
    layers.Dense(10, activation = 'softmax') #Output layer with 10 digits
])

#Compile the model

model.compile(optimizer='adam',
              loss = 'sparse_categorical_crossentropy',
              metrics = ['accuracy'])

#Train the model

model.fit(train_images, train_labels, epochs=5, batch_size=32)

#Evaluate the model

test_loss, test_acc = model.evaluate(test_images, test_labels)
print(f'Test accuracy: {test_acc:.4f}')

#Make predictions

predictions = model.predict(test_images)

# Visualize some predictions
def plot_image(i, predictions_array, true_label, img):
    predictions_array, true_label, img = predictions_array[i], true_label[i], img[i].reshape(28, 28)
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])

    plt.imshow(img, cmap=plt.cm.binary)

    predicted_label = np.argmax(predictions_array)
    color = 'blue' if predicted_label == true_label else 'red'

    plt.xlabel(f"{predicted_label} ({100*np.max(predictions_array):2.0f}%)",
               color=color)

def plot_value_array(i, predictions_array, true_label):
    predictions_array, true_label = predictions_array[i], true_label[i]
    plt.grid(False)
    plt.xticks(range(10))
    plt.yticks([])
    plt.bar(range(10), predictions_array, color="#777777")
    plt.ylim([0, 1])
    predicted_label = np.argmax(predictions_array)

    plt.xticks(range(10), [f'{i}' for i in range(10)])
    plt.xticks(range(10), [f'{i}' for i in range(10)])

    plt.bar(predicted_label, predictions_array[predicted_label], color='red')
    plt.bar(true_label, predictions_array[true_label], color='blue')

# Plot a few test images and predictions
num_rows = 5
num_cols = 2
num_images = num_rows * num_cols
plt.figure(figsize=(2 * 2 * num_cols, 2 * num_rows))
for i in range(num_images):
    plt.subplot(num_rows, 2 * num_cols, 2 * i + 1)
    plot_image(i, predictions, test_labels, test_images)
    plt.subplot(num_rows, 2 * num_cols, 2 * i + 2)
    plot_value_array(i, predictions, test_labels)
plt.tight_layout()
plt.show()