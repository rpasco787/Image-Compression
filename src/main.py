from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

def image_to_matrix(image_path):
    """
    Convert a grayscale image to a matrix.
    
    Inputs:
        image_path (str): Path to the input image file
        
    Returns:
        numpy.ndarray: 2D matrix representing the grayscale image
    """
    # Load the image
    image = Image.open(image_path)
    # Convert to grayscale if it's not already
    if image.mode != 'L':
        image = image.convert('L')
    # Convert PIL image to numpy array (matrix)
    matrix = np.array(image)
    return matrix

def display_image(matrix, title="Grayscale Image"):
    """
    Display the image matrix using matplotlib.
    
    Inputs:
        matrix (numpy.ndarray): 2D matrix representing the grayscale image
        title (str): Title for the displayed image
    """
    plt.figure(figsize=(8, 6))
    plt.imshow(matrix, cmap='gray')
    plt.title(title)
    plt.axis('off')  # Hide axes for cleaner display
    plt.show()

def main():
    # Example usage
    image_path = input("Enter the path to your grayscale image: ")
    
    try:
        image = image_to_matrix(image_path)
        U, S, Vt = np.linalg.svd(image, full_matrices=False)
        print(S.shape)

    except FileNotFoundError:
        print(f"Error: Could not find the image file at '{image_path}'")
    except Exception as e:
        print(f"Error processing image: {e}")

if __name__ == '__main__':
    main()