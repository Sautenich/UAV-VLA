"""
Coordinate Transformation Module

This module provides functionality to transform coordinates between percentage-based
and latitude/longitude formats using reference points from satellite images.
"""

import pandas as pd
from typing import Dict, Tuple, Any, List
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def read_coordinates_from_csv(file_path: str) -> Dict[str, Tuple[float, float, float, float]]:
    """
    Read corner coordinates from a CSV file.
    
    Args:
        file_path: Path to the CSV file containing corner coordinates
        
    Returns:
        Dictionary mapping image names to corner coordinates (NW lat, NW lon, SE lat, SE lon)
        
    Raises:
        FileNotFoundError: If the CSV file doesn't exist
        pd.errors.EmptyDataError: If the CSV file is empty
    """
    try:
        df = pd.read_csv(file_path)
        coordinates_dict = {}
        
        for _, row in df.iterrows():
            image_name = row['Image']
            nw_lat = float(row['NW Corner Lat'])
            nw_lon = float(row['NW Corner Long'])
            se_lat = float(row['SE Corner Lat'])
            se_lon = float(row['SE Corner Long'])
            
            coordinates_dict[image_name] = (nw_lat, nw_lon, se_lat, se_lon)
            
        return coordinates_dict
        
    except FileNotFoundError:
        logger.error(f"CSV file not found: {file_path}")
        raise
    except pd.errors.EmptyDataError:
        logger.error(f"CSV file is empty: {file_path}")
        raise
    except Exception as e:
        logger.error(f"Error reading CSV file: {e}")
        raise

def percentage_to_lat_lon(
    x_percent: float,
    y_percent: float,
    nw_lat: float,
    nw_lon: float,
    se_lat: float,
    se_lon: float
) -> Tuple[float, float]:
    """
    Convert percentage-based coordinates to latitude/longitude.
    
    Args:
        x_percent: X coordinate as percentage (0-100)
        y_percent: Y coordinate as percentage (0-100)
        nw_lat: Northwest corner latitude
        nw_lon: Northwest corner longitude
        se_lat: Southeast corner latitude
        se_lon: Southeast corner longitude
        
    Returns:
        Tuple of (latitude, longitude)
    """
    try:
        # Calculate the total span of latitude and longitude
        lat_span = nw_lat - se_lat
        lon_span = se_lon - nw_lon
        
        # Convert percentages to actual coordinates
        lat = nw_lat - (y_percent / 100.0 * lat_span)
        lon = nw_lon + (x_percent / 100.0 * lon_span)
        
        return lat, lon
        
    except Exception as e:
        logger.error(f"Error converting percentage to lat/lon: {e}")
        raise

def recalculate_coordinates(
    objects_json: Dict[str, Dict[str, Any]],
    image_number: int,
    coordinates_dict: Dict[str, Tuple[float, float, float, float]]
) -> Dict[str, Dict[str, Any]]:
    """
    Recalculate percentage-based coordinates to latitude/longitude for all objects.
    
    Args:
        objects_json: Dictionary of objects with percentage-based coordinates
        image_number: Image number to use for reference coordinates
        coordinates_dict: Dictionary of corner coordinates for each image
        
    Returns:
        Dictionary of objects with latitude/longitude coordinates
        
    Raises:
        KeyError: If image number not found in coordinates dictionary
        ValueError: If coordinates are invalid
    """
    try:
        # Get corner coordinates for the specified image
        if str(image_number) not in coordinates_dict:
            raise KeyError(f"Image number {image_number} not found in coordinates dictionary")
            
        nw_lat, nw_lon, se_lat, se_lon = coordinates_dict[str(image_number)]
        
        # Initialize output dictionary
        result_coordinates = {}
        
        # Process each object
        for obj_id, obj_data in objects_json.items():
            try:
                obj_type = obj_data["type"]
                x_percent, y_percent = obj_data["coordinates"]
                
                # Convert to lat/lon
                lat, lon = percentage_to_lat_lon(
                    x_percent, y_percent,
                    nw_lat, nw_lon,
                    se_lat, se_lon
                )
                
                result_coordinates[obj_id] = {
                    "type": obj_type,
                    "coordinates": [lat, lon]
                }
                
            except (KeyError, ValueError, TypeError) as e:
                logger.warning(f"Error processing object {obj_id}: {e}")
                continue
                
        return result_coordinates
        
    except Exception as e:
        logger.error(f"Error recalculating coordinates: {e}")
        raise

if __name__ == "__main__":
    # Example usage
    example_objects = {
        "building_1": {"type": "building", "coordinates": [40.2, 39.5]},
        "building_2": {"type": "building", "coordinates": [47.7, 39.0]},
        "building_3": {"type": "building", "coordinates": [64.9, 41.2]},
        "building_4": {"type": "building", "coordinates": [65.2, 87.9]},
        "building_5": {"type": "building", "coordinates": [80.2, 20.7]}
    }
    
    try:
        # Read coordinates from CSV
        csv_file_path = 'benchmark-UAV-VLPA-nano-30/parsed_coordinates.csv'
        coordinates_dict = read_coordinates_from_csv(csv_file_path)
        
        # Example: recalculate coordinates for image 38
        image_number = 38
        result = recalculate_coordinates(example_objects, image_number, coordinates_dict)
        
        logger.info("Successfully recalculated coordinates:")
        for obj_id, obj_data in result.items():
            logger.info(f"{obj_id}: {obj_data}")
            
    except Exception as e:
        logger.error(f"Error in example: {e}")