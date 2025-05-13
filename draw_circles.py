"""
Image Visualization Module

This module provides functionality to visualize identified points and paths
on satellite imagery by drawing circles and lines.
"""

import cv2
import numpy as np
from typing import Dict, Any, List, Tuple, Optional
import logging
import os

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def draw_dots_and_lines_on_image(
    image_path: str,
    points_dict: Dict[str, Dict[str, Any]],
    output_path: str,
    circle_radius: int = 5,
    circle_color: Tuple[int, int, int] = (0, 0, 255),  # Red in BGR
    line_color: Tuple[int, int, int] = (0, 255, 0),    # Green in BGR
    thickness: int = 2
) -> Optional[np.ndarray]:
    """
    Draw circles at specified points and connect them with lines on an image.
    
    Args:
        image_path: Path to the input image
        points_dict: Dictionary containing points with their coordinates
        output_path: Path to save the output image
        circle_radius: Radius of the circles to draw
        circle_color: Color of the circles in BGR format
        line_color: Color of the connecting lines in BGR format
        thickness: Thickness of circles and lines
        
    Returns:
        Modified image as numpy array, or None if processing fails
        
    Raises:
        FileNotFoundError: If input image doesn't exist
        ValueError: If points dictionary is empty or invalid
        Exception: For other processing errors
    """
    try:
        # Validate input
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")
            
        if not points_dict:
            raise ValueError("Points dictionary is empty")
            
        # Read the image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Failed to read image: {image_path}")
            
        # Get image dimensions
        height, width = image.shape[:2]
        
        # Extract and validate points
        points = []
        for point_data in points_dict.values():
            try:
                x, y = point_data["coordinates"]
                # Convert percentage to pixel coordinates
                x_pixel = int((x / 100.0) * width)
                y_pixel = int((y / 100.0) * height)
                points.append((x_pixel, y_pixel))
            except (KeyError, ValueError, TypeError) as e:
                logger.warning(f"Invalid point data: {point_data}, Error: {e}")
                continue
                
        if not points:
            raise ValueError("No valid points found in points dictionary")
            
        # Draw circles at each point
        for point in points:
            cv2.circle(
                image,
                point,
                circle_radius,
                circle_color,
                thickness
            )
            
        # Draw lines connecting points in sequence
        for i in range(len(points) - 1):
            cv2.line(
                image,
                points[i],
                points[i + 1],
                line_color,
                thickness
            )
            
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Save the image
        cv2.imwrite(output_path, image)
        logger.info(f"Saved annotated image to: {output_path}")
        
        return image
        
    except Exception as e:
        logger.error(f"Error processing image: {e}")
        return None

if __name__ == "__main__":
    # Example usage
    example_points = {
        "point1": {"type": "building", "coordinates": [25.0, 25.0]},
        "point2": {"type": "building", "coordinates": [50.0, 50.0]},
        "point3": {"type": "building", "coordinates": [75.0, 75.0]}
    }
    
    try:
        # Test with a sample image
        test_image = "benchmark-UAV-VLPA-nano-30/images/1.jpg"
        output_image = "test_output.jpg"
        
        result = draw_dots_and_lines_on_image(
            test_image,
            example_points,
            output_image
        )
        
        if result is not None:
            logger.info("Successfully processed test image")
        else:
            logger.error("Failed to process test image")
            
    except Exception as e:
        logger.error(f"Error in example: {e}")