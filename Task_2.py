import numpy as np
import cv2
import matplotlib.pyplot as plt
from collections import deque

def add_gaussian_noise(image, mean=0, sigma=25):
    """Add Gaussian noise to the image"""
    gauss = np.random.normal(mean, sigma, image.shape)
    noisy_image = image + gauss
    noisy_image = np.clip(noisy_image, 0, 255).astype(np.uint8)
    return noisy_image

def region_growing(image, seeds, threshold):
    """
    Region growing algorithm for image segmentation
    
    Parameters:
    - image: Input image (grayscale)
    - seeds: List of seed points (x, y)
    - threshold: Maximum difference between seed and neighbor pixel values
    
    Returns:
    - Binary mask of segmented region
    """
    # Output mask
    height, width = image.shape
    segmented = np.zeros((height, width), dtype=np.uint8)
    
    # 8-connectivity neighbors
    neighbors = [(-1, -1), (-1, 0), (-1, 1),
                 (0, -1),          (0, 1),
                 (1, -1),  (1, 0), (1, 1)]
    
    # Queue for pixels to process
    queue = deque()
    
    # Add seeds to queue and mark them in segmented image
    for seed in seeds:
        x, y = seed
        segmented[y, x] = 1
        queue.append((y, x))
    
    # Get seed value (average of all seeds)
    seed_value = np.mean([image[y, x] for x, y in seeds])
    
    # Process pixels
    while queue:
        y, x = queue.popleft()
        
        # Check all 8 neighbors
        for dy, dx in neighbors:
            ny, nx = y + dy, x + dx
            
            # Check if neighbor is within image bounds
            if 0 <= ny < height and 0 <= nx < width:
                # Check if neighbor is not already segmented
                if segmented[ny, nx] == 0:
                    # Check if neighbor is within threshold
                    if abs(int(image[ny, nx]) - int(seed_value)) <= threshold:
                        segmented[ny, nx] = 1
                        queue.append((ny, nx))
    
    return segmented

def test_region_growing():
    # Create a test image (circle on background)
    image_path = r"C:\Users\rpras\OneDrive\Documents\Rashmitha\Semester_7\Vision\Assignment2\Asserts\Lemmon.png"
    
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    if image is None:
        print(f"Error: Could not load image from {image_path}")
        return
    
    # Add some noise
    noisy_image = add_gaussian_noise(image, sigma=20)
    
    # Define seeds (points inside the circle)
    seeds = [(128, 128), (120, 120), (135, 135)]
    
    # Set threshold
    threshold = 30
    
    # Apply region growing
    segmented = region_growing(noisy_image, seeds, threshold)
    
    # Display results
    plt.figure(figsize=(5, 5))
    plt.imshow(image, cmap='gray')
    plt.title('Original Image')
    plt.axis('off')
    
    plt.figure(figsize=(5, 5))
    plt.imshow(noisy_image, cmap='gray')
    plt.title('Noisy Image')
    plt.axis('off')
    
    plt.figure(figsize=(5, 5))
    plt.imshow(segmented, cmap='gray')
    plt.title('Region Growing Segmentation')
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()

# Run the test
test_region_growing()