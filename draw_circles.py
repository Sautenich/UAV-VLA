import json
import os
from PIL import Image, ImageDraw
from config import *

def draw_dots_and_lines_on_image(image_path, json_data, output_folder='identified_images', output_path='sample_pic0.jpg'):
    # Open the image
    image = Image.open(image_path)
    draw = ImageDraw.Draw(image)
    
    # Get image dimensions
    width, height = image.size

    # Define colors for the dots and lines
    dot_color = (128, 0, 128)  # Purple
    line_color = (255, 255, 0)  # Yellow
    dot_size = 5  # Size of the dot

    # Store pixel coordinates for line drawing
    pixel_coordinates = []

    # Iterate through the JSON data
    for building, data in json_data.items():
        # Extract coordinates
        coordinates = data['coordinates']
        x_percent = coordinates[0]
        y_percent = coordinates[1]
        
        # Convert percentage to pixel values
        x_pixel = (x_percent / 100) * width
        y_pixel = (y_percent / 100) * height
        
        # Draw a dot on the image
        draw.ellipse((x_pixel - dot_size, y_pixel - dot_size, x_pixel + dot_size, y_pixel + dot_size), fill=dot_color)
        
        # Store the pixel coordinates for line drawing
        pixel_coordinates.append((x_pixel, y_pixel))

    # Draw lines connecting the dots
    for i in range(len(pixel_coordinates) - 1):
        draw.line([pixel_coordinates[i], pixel_coordinates[i + 1]], fill=line_color, width=2)

    # Create output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Save the modified image
    #output_path = os.path.join(output_folder, 'output_image33.png')
    image.save(output_path)
    print(f"Image saved to {output_path}")


def main():
    # Example usage
    image_path = 'new_data/1.jpg'  # Replace with your image path
    draw_dots_and_lines_on_image(image_path, json_data, output_path='output_image33.png')

if __name__=='__main__':
    main()