import numpy as np
import cv2
import matplotlib.pyplot as plt

def add_gaussian_noise(image, mean=0, sigma=25):
    """Add Gaussian noise to the image"""
    gauss = np.random.normal(mean, sigma, image.shape)
    noisy_image = image + gauss
    noisy_image = np.clip(noisy_image, 0, 255).astype(np.uint8)
    return noisy_image

def otsu_thresholding(image):
    """Implement Otsu's thresholding algorithm"""
    # Calculate histogram
    hist = cv2.calcHist([image], [0], None, [256], [0, 256])
    hist_norm = hist.ravel() / hist.sum()
    
    # Initialize variables
    best_threshold = 0
    max_variance = 0
    
    # Try all possible thresholds
    for threshold in range(256):
        # Class probabilities
        w0 = np.sum(hist_norm[:threshold])
        w1 = np.sum(hist_norm[threshold:])
        
        if w0 == 0 or w1 == 0:
            continue
            
        # Class means
        mean0 = np.sum(np.arange(threshold) * hist_norm[:threshold]) / w0
        mean1 = np.sum(np.arange(threshold, 256) * hist_norm[threshold:]) / w1
        
        # Between-class variance
        variance = w0 * w1 * (mean0 - mean1) ** 2
        
        if variance > max_variance:
            max_variance = variance
            best_threshold = threshold
    
    # Apply threshold
    _, binary = cv2.threshold(image, best_threshold, 255, cv2.THRESH_BINARY)
    return binary, best_threshold

def test_otsu_algorithm():
    # Loading image
    image_path = r"C:\Users\rpras\OneDrive\Documents\Rashmitha\Semester_7\Vision\Assignment2\Asserts\Apple_Orange.jpg"
    
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    if image is None:
        print(f"Error: Could not load image from {image_path}")
        return
    
    
    # Add Gaussian noise
    noisy_image = add_gaussian_noise(image)
    
    # Apply Otsu's algorithm
    binary, threshold = otsu_thresholding(noisy_image)
    
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
    plt.imshow(binary, cmap='gray')
    plt.title(f'Otsu Thresholding (Threshold: {threshold})')
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()

# Run the test
test_otsu_algorithm()