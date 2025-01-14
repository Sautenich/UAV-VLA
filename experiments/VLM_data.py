import re

def extract_coordinates(data):
    coords = {}
    current_img = None
    for line in data.splitlines():
        if 'Image' in line:
            current_img = int(re.search(r'Image: (\d+)', line).group(1))
            coords[current_img] = {}
        elif 'NW Corner Lat' in line:
            coords[current_img]['NW_Lat'] = float(re.search(r'NW Corner Lat: ([\d\-.]+)', line).group(1))
        elif 'NW Corner Long' in line:
            coords[current_img]['NW_Long'] = float(re.search(r'NW Corner Long: ([\d\-.]+)', line).group(1))
        elif 'SE Corner Lat' in line:
            coords[current_img]['SE_Lat'] = float(re.search(r'SE Corner Lat: ([\d\-.]+)', line).group(1))
        elif 'SE Corner Long' in line:
            coords[current_img]['SE_Long'] = float(re.search(r'SE Corner Long: ([\d\-.]+)', line).group(1))
    return coords

def extract_data_points(data):
    points = {}
    for line in data.splitlines():
        match = re.match(r'(\d+),\s*<points (.*?)</points>', line)
        if match:
            img_num = int(match.group(1))
            points_str = match.group(2)
            coords = re.findall(r'x\d+="([\d.]+)"\s*y\d+="([\d.]+)"', points_str)
            points[img_num] = [(float(x), float(y)) for x, y in coords]
    return points

def extract_home_positions(data):
    home_positions = {}
    for line in data.splitlines():
        if 'Image' in line:
            img_num = int(re.search(r'Image (\d+)', line).group(1))
            home_positions[img_num] = {}
        elif 'Latitude' in line:
            home_positions[img_num]['Latitude'] = float(re.search(r'Latitude = ([\d\-.]+)', line).group(1))
        elif 'Longitude' in line:
            home_positions[img_num]['Longitude'] = float(re.search(r'Longitude = ([\d\-.]+)', line).group(1))
    return home_positions

# Convert normalized points to lat/long
def convert_to_latlong(image_num, points, latlong_data):
    NW_Lat = latlong_data[image_num]['NW_Lat']
    NW_Long = latlong_data[image_num]['NW_Long']
    SE_Lat = latlong_data[image_num]['SE_Lat']
    SE_Long = latlong_data[image_num]['SE_Long']
    
    latlong_points = []
    for x, y in points:
        lat = NW_Lat - (y / 100) * (NW_Lat - SE_Lat)
        long = NW_Long + (x / 100) * (SE_Long - NW_Long)
        latlong_points.append((lat, long))
    
    return latlong_points

def process_vlm_data(latlong_file, stadium_file, home_position_file, output_file):
    with open(latlong_file, 'r') as file:
        latlong_data = extract_coordinates(file.read())

    with open(stadium_file, 'r') as file:
        data_points = extract_data_points(file.read())

    with open(home_position_file, 'r') as file:
        home_positions = extract_home_positions(file.read())

    with open(output_file, 'w') as f:
        for img_num, points in data_points.items():
            if img_num in home_positions:
                f.write(f"Image {img_num}:\n")
                f.write("  Home Position:\n")
                f.write(f"    Latitude: {home_positions[img_num]['Latitude']}\n")
                f.write(f"    Longitude: {home_positions[img_num]['Longitude']}\n")
            
            latlong_points = convert_to_latlong(img_num, points, latlong_data)
            for i, (lat, long) in enumerate(latlong_points, start=1):
                f.write(f"  Building {i}:\n")
                f.write(f"    Latitude: {lat}\n")
                f.write(f"    Longitude: {long}\n")
            f.write("\n")

    print(f"All coordinates with home positions saved to {output_file}")
