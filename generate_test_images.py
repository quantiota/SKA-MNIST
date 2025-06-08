import matplotlib.pyplot as plt
import torch
import torchvision
import torchvision.transforms as transforms

# Load MNIST dataset using PyTorch
transform = transforms.Compose([transforms.ToTensor()])
test_dataset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)

# Set random seed for reproducibility (optional)
torch.manual_seed(42)

# Get 100 random indices from the test set
test_size = len(test_dataset)
random_indices = torch.randperm(test_size)[:100]

# Extract the 100 random images and labels
random_images = torch.stack([test_dataset[i][0] for i in random_indices])
random_labels = torch.tensor([test_dataset[i][1] for i in random_indices])

# Remove channel dimension (MNIST is grayscale)
random_images = random_images.squeeze(1)

print(f"Selected 100 random images from MNIST test set")
print(f"Image shape: {random_images[0].shape}")
print(f"Labels: {random_labels}")
print(f"Label distribution: {torch.bincount(random_labels)}")

# Display first 25 images in a 5x5 grid
plt.figure(figsize=(12, 12))
for i in range(25):
    plt.subplot(5, 5, i + 1)
    plt.imshow(random_images[i], cmap='gray')
    plt.title(f'Label: {random_labels[i].item()}')
    plt.axis('off')

plt.suptitle('First 25 Random MNIST Test Images', fontsize=16)
plt.tight_layout()
plt.show()

# You can access individual images like this:
# random_images[0] - first random image (28x28 tensor)
# random_labels[0] - corresponding label

# If you want to save the tensors:
# torch.save(random_images, 'random_mnist_images.pt')
# torch.save(random_labels, 'random_mnist_labels.pt')

# Example: Print some statistics
print(f"\nDataset statistics:")
print(f"Number of images: {len(random_images)}")
print(f"Image dimensions: {random_images.shape}")
print(f"Pixel value range: {random_images.min():.3f} to {random_images.max():.3f}")
print(f"Data type: {random_images.dtype}")
print(f"Labels dtype: {random_labels.dtype}")