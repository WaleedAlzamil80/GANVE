import  torch
import matplotlib.pyplot as plt

# Plot original and reconstructed distributions for comparison
def plot_images(original, reconstructed, num_samples=5):
    fig, axes = plt.subplots(num_samples, 2, figsize=(10, 10))
    for i in range(num_samples):
        axes[i, 0].imshow(original[i].reshape(28, 28), cmap='gray')
        axes[i, 0].set_title("Original")
        axes[i, 0].axis('off')

        axes[i, 1].imshow(reconstructed[i].reshape(28, 28), cmap='gray')
        axes[i, 1].set_title("Reconstructed")
        axes[i, 1].axis('off')
    plt.tight_layout()
    plt.show()

# Collect original and reconstructed data for comparison
def compare(model, data):
    original_data = []
    reconstructed_data = []

    with torch.no_grad():
        for images, _ in data:
            original_data.append((images.cpu().view(-1, 784) > 0.5).float())
            reconstructed_data.append((model((images.view(-1, 784) > 0.5).float().to('cuda')).cpu() > 0.5).float())

    original_data = torch.cat(original_data, dim=0).numpy()
    reconstructed_data = torch.cat(reconstructed_data, dim=0).numpy()
    return original_data, reconstructed_data

def plot_pixel_value_distributions(original, reconstructed):
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    # Sum the number of times each pixel is 1 across all images
    original_sum = original.sum(axis=0)
    reconstructed_sum = reconstructed.sum(axis=0)

    # Plot the original pixel values
    axes[0].plot(range(len(original_sum)), original_sum, color='blue', alpha=0.7)
    axes[0].set_title("Original Pixel Values")
    axes[0].set_xlabel("Pixel Index")
    axes[0].set_ylabel("Pixel Value")

    # Plot the reconstructed pixel values
    axes[1].plot(range(len(reconstructed_sum)), reconstructed_sum, color='red', alpha=0.7)
    axes[1].set_title("Reconstructed Pixel Values")
    axes[1].set_xlabel("Pixel Index")
    axes[1].set_ylabel("Pixel Value")

    plt.tight_layout()
    plt.show()


def plot_pixel_value_distributions_3d(original, reconstructed):
    fig = plt.figure(figsize=(15, 10))

    # Reshape the data assuming each image is 28x28
    original_sum = original.sum(axis=0).reshape(28, 28)
    reconstructed_sum = reconstructed.sum(axis=0).reshape(28, 28)

    # Create 3D plot for original pixel values
    ax1 = fig.add_subplot(121, projection='3d')
    x, y = np.meshgrid(range(28), range(28))
    ax1.plot_surface(x, y, original_sum, cmap='viridis')
    ax1.set_title("Original Pixel Values")
    ax1.set_xlabel("X Pixel Index")
    ax1.set_ylabel("Y Pixel Index")
    ax1.set_zlabel("Pixel Value")

    # Create 3D plot for reconstructed pixel values
    ax2 = fig.add_subplot(122, projection='3d')
    ax2.plot_surface(x, y, reconstructed_sum, cmap='viridis')
    ax2.set_title("Reconstructed Pixel Values")
    ax2.set_xlabel("X Pixel Index")
    ax2.set_ylabel("Y Pixel Index")
    ax2.set_zlabel("Pixel Value")

    plt.tight_layout()
    plt.show()