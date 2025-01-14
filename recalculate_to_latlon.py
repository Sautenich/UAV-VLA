import pandas as pd

def read_coordinates_from_csv(file_path):
    """Reads the coordinates from the CSV file and returns a dictionary."""
    df = pd.read_csv(file_path)
    coordinates_dict = {}
    
    for _, row in df.iterrows():
        image_name = row['Image']
        nw_lat = row['NW Corner Lat']
        nw_lon = row['NW Corner Long']
        se_lat = row['SE Corner Lat']
        se_lon = row['SE Corner Long']
        
        coordinates_dict[image_name] = (nw_lat, nw_lon, se_lat, se_lon)
    
    return coordinates_dict

def percentage_to_lat_lon(nw_lat, nw_lon, se_lat, se_lon, percentage_x, percentage_y):
    """Converts percentage coordinates (scaled by 100) to geographic coordinates."""
    x = percentage_x / 100.0
    y = percentage_y / 100.0
    
    lat_range = nw_lat - se_lat
    lon_range = se_lon - nw_lon
    
    lat = nw_lat - (lat_range * y)
    lon = nw_lon + (lon_range * x)
    
    return lat, lon

def recalculate_coordinates(objects_json, image_number, coordinates_dict):
    """Recalculates latitude and longitude based on percentage coordinates (scaled by 100)."""
    result = {}
    
    image_name = f"{image_number}.jpg"  # Assuming the provided number corresponds to the image
    if image_name in coordinates_dict:
        nw_lat, nw_lon, se_lat, se_lon = coordinates_dict[image_name]
        
        for key, value in objects_json.items():
            percentage_x, percentage_y = value["coordinates"]  # Assuming coordinates are [percentage_x, percentage_y]
            
            lat, lon = percentage_to_lat_lon(nw_lat, nw_lon, se_lat, se_lon, percentage_x, percentage_y)
            result[key] = {
                "type": value["type"],
                "coordinates": [lat, lon]
            }
    
    return result


if __name__ == "__main__":
    # Example usage
    objects_json = {
        "building_1": {"type": "building", "coordinates": [40.2, 39.5]},  # Example percentage coordinates
        "building_2": {"type": "building", "coordinates": [47.7, 39.0]},
        "building_3": {"type": "building", "coordinates": [64.9, 41.2]},
        "building_4": {"type": "building", "coordinates": [65.2, 87.9]},
        "building_5": {"type": "building", "coordinates": [80.2, 20.7]}
    }

    # Path to the CSV file
    csv_file_path = 'parsed_coordinates.csv'
    coordinates_dict = read_coordinates_from_csv(csv_file_path)

    # Specify the image number you want to use for recalculation
    image_number = 38 # Example: using image number 1
    result_coordinates = recalculate_coordinates(objects_json, image_number, coordinates_dict)

    print(result_coordinates)