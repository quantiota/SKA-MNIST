import matplotlib.pyplot as plt
import torch
import torchvision
import torchvision.transforms as transforms
from PIL import Image
import os

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

# Create directory for saving images
output_dir = './mnist_test_images'
os.makedirs(output_dir, exist_ok=True)

# Save each image as a PNG file
print(f"\nSaving images to {output_dir}/")
for i in range(100):
    # Convert tensor to PIL Image (scale from [0,1] to [0,255])
    img_array = (random_images[i] * 255).byte().numpy()
    img = Image.fromarray(img_array, mode='L')  # 'L' for grayscale
    
    # Save with filename: image_001_label_7.png (index_label)
    filename = f"image_{i:03d}_label_{random_labels[i].item()}.png"
    filepath = os.path.join(output_dir, filename)
    img.save(filepath)
    
    if i < 5:  # Print first 5 filenames as example
        print(f"  {filename}")

print(f"✓ Saved all 100 images to {output_dir}/")
print(f"  Filename format: image_XXX_label_Y.png")
print(f"  Where XXX is the index (000-099) and Y is the digit (0-9)")

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

# You can also access the tensors:
# random_images[0] - first random image (28x28 tensor)
# random_labels[0] - corresponding label

# Example: Print some statistics
print(f"\nDataset statistics:")
print(f"Number of images: {len(random_images)}")
print(f"Image dimensions: {random_images.shape}")
print(f"Pixel value range: {random_images.min():.3f} to {random_images.max():.3f}")
print(f"Data type: {random_images.dtype}")
print(f"Labels dtype: {random_labels.dtype}")