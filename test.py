import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from minisom import MiniSom
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.metrics import mutual_info_score, adjusted_rand_score


def load_image(filename):
    img = Image.open(filename)
    img = img.convert("RGB")  # Convert image to RGB format
    img_array = np.array(img)
    return img, img_array

def compress_image(input_image, som_shape=(8, 8), num_iterations=1000):
    input_shape = input_image.shape
    input_data = input_image.reshape(-1, 3)  # Flatten image into vectors

    som = MiniSom(som_shape[0], som_shape[1], 3, sigma=0.5, learning_rate=0.5)  # Initialize SOM

    som.random_weights_init(input_data)  # Initialize SOM weights
    som.train_random(input_data, num_iterations)  # Train SOM

    compressed_image = np.zeros(input_data.shape)

    for i, vec in enumerate(input_data):
        winner = som.winner(vec)  # Find the winning neuron for each input vector
        compressed_image[i] = som._weights[winner]  # Replace input vector with SOM weight vector of the winning neuron

    compressed_image = compressed_image.reshape(input_shape)  # Reshape compressed image

    return compressed_image.astype(np.uint8), som

def save_image(image_array, filename):
    img = Image.fromarray(image_array)
    img.save(filename)

def get_file_size_in_kb(filename):
    return os.path.getsize(filename) / 1024  # Convert bytes to kilobytes

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
    input_filename = "img.jpg"  # Replace with your input image filename
    output_filename = "compressed_image.jpg"  # Replace with your desired output image filename

    original_image, input_image_array = load_image(input_filename)
    compressed_image, _ = compress_image(input_image_array)

    print(_.winner().count())

    save_image(compressed_image, output_filename)

    original_size_kb = get_file_size_in_kb(input_filename)
    compressed_size_kb = get_file_size_in_kb(output_filename)

    print("Image compression complete. Compressed image saved as", output_filename)

    display_images_with_size(original_image, compressed_image, original_size_kb, compressed_size_kb)
