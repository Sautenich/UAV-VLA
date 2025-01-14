import math
import re

def calculate_offset(nw_lat, nw_long, se_lat, se_long, x_percent, y_percent):
    """Calculate new coordinates based on given percentage offset."""
    x_offset = x_percent / 100
    y_offset = y_percent / 100
    
    new_lat = nw_lat - (nw_lat - se_lat) * y_offset
    new_long = nw_long + (se_long - nw_long) * x_offset
    
    return new_lat, new_long

def process_image_data(input_file, output_file):
    """Process image data and extract home position and coordinates."""
    with open(input_file, 'r') as file, open(output_file, 'w') as output:
        current_image = None
        nw_lat, nw_long, se_lat, se_long = None, None, None, None
        for line in file:
            line = line.strip()
            if line.startswith("Image:"):
                if current_image and nw_lat and nw_long and se_lat and se_long:
                    # Calculate the coordinates at 10% x and 10% y offset
                    new_lat, new_long = calculate_offset(nw_lat, nw_long, se_lat, se_long, 10, 10)
                    output.write(f"\nImage {current_image}:\n  Home Position:\n    Latitude = {new_lat}\n    Longitude = {new_long}\n")

                current_image = line.split(":")[1].strip()
                nw_lat, nw_long, se_lat, se_long = None, None, None, None
            elif line.startswith("NW Corner Lat:"):
                nw_lat = float(line.split(":")[1].strip().rstrip(','))
            elif line.startswith("NW Corner Long:"):
                nw_long = float(line.split(":")[1].strip().rstrip(','))
            elif line.startswith("SE Corner Lat:"):
                se_lat = float(line.split(":")[1].strip().rstrip(','))
            elif line.startswith("SE Corner Long:"):
                se_long = float(line.split(":")[1].strip().rstrip(','))
        
        if current_image and nw_lat and nw_long and se_lat and se_long:
            new_lat, new_long = calculate_offset(nw_lat, nw_long, se_lat, se_long, 10, 10)
            output.write(f"\nImage {current_image}:\n  Home Position:\n    Latitude = {new_lat}\n    Longitude = {new_long}\n")

    print(f"Coordinates saved to {output_file}")
