import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os

class SOM:
    def __init__(self, input_dim, output_dim, learning_rate=0.1, sigma=1.0):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.learning_rate = learning_rate
        self.sigma = sigma
        self.weights = np.random.rand(output_dim[0], output_dim[1], input_dim)

    def update_weights(self, input_vector, winner):
        for i in range(self.output_dim[0]):
            for j in range(self.output_dim[1]):
                distance = np.linalg.norm(np.array([i, j]) - np.array(winner))
                influence = np.exp(-(distance ** 2) / (2 * (self.sigma ** 2)))
                self.weights[i, j] += self.learning_rate * influence * (input_vector - self.weights[i, j])

    def train(self, input_data, num_iterations):
        for _ in range(num_iterations):
            for input_vector in input_data:
                winner = self.find_winner(input_vector)
                self.update_weights(input_vector, winner)
            self.learning_rate *= 0.99  # Decay learning rate
            self.sigma *= 0.99  # Decay sigma

    def find_winner(self, input_vector):
        min_distance = float('inf')
        winner = None
        for i in range(self.output_dim[0]):
            for j in range(self.output_dim[1]):
                distance = np.linalg.norm(input_vector - self.weights[i, j])
                if distance < min_distance:
                    min_distance = distance
                    winner = (i, j)
        return winner

def load_image(filename):
    img = Image.open(filename)
    img = img.convert("RGB")  # Convert image to RGB format
    img_array = np.array(img)
    return img, img_array

def compress_image(input_image, som_shape=(8, 8), num_iterations=3):
    input_shape = input_image.shape
    input_data = input_image.reshape(-1, 3)  # Flatten image into vectors

    som = SOM(input_data.shape[1], som_shape)

    som.train(input_data, num_iterations)

    compressed_image = np.zeros(input_data.shape)

    for i, vec in enumerate(input_data):
        winner = som.find_winner(vec)  # Find the winning neuron for each input vector
        compressed_image[i] = som.weights[winner]  # Replace input vector with SOM weight vector of the winning neuron

    compressed_image = compressed_image.reshape(input_shape)  # Reshape compressed image

    return compressed_image.astype(np.uint8)

def save_image(image_array, filename):
    img = Image.fromarray(image_array)
    img.save(filename)

def display_images_with_size(original_img, compressed_img, original_size_kb, compressed_size_kb):
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(original_img)
    axes[0].set_title(f"Original Image\nSize: {original_size_kb:.2f} KB")
    axes[0].axis('off')
    axes[1].imshow(compressed_img)
    axes[1].set_title(f"Compressed Image\nSize: {compressed_size_kb:.2f} KB")
    axes[1].axis('off')
    plt.show()

if __name__ == "__main__":
    input_filename = "1.png"  # Replace with your input image filename
    output_filename = "compressed_image.jpg"  # Replace with your desired output image filename

    original_image, input_image_array = load_image(input_filename)
    compressed_image = compress_image(input_image_array)

    save_image(compressed_image, output_filename)

    original_size_kb = os.path.getsize(input_filename) / 1024
    compressed_size_kb = os.path.getsize(output_filename) / 1024

    print("Image compression complete. Compressed image saved as", output_filename)

    display_images_with_size(original_image, compressed_image, original_size_kb, compressed_size_kb)
