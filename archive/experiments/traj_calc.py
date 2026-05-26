import math
import numpy as np

# Convert latitude and longitude differences to kilometers
def lat_lon_to_km(lat, lon):
    lat_m = lat * 111.320  # 1 degree latitude = 111,320 meters
    lon_m = lon * 111.320 * np.cos(np.radians(lat))  # Adjust longitude by latitude
    return lat_m, lon_m

def euclidean_distance_km(lat1, lon1, lat2, lon2):
    lat1_m, lon1_m = lat_lon_to_km(lat1, lon1)
    lat2_m, lon2_m = lat_lon_to_km(lat2, lon2)
    return np.sqrt((lat2_m - lat1_m)**2 + (lon2_m - lon1_m)**2)

def calculate_trajectory(file_path):
    """Calculate the total trajectory length for a given file."""
    with open(file_path, 'r') as file:
        total_trajectory_lengths = {}
        current_image = None
        coordinates = []

        for line in file:
            line = line.strip()

            if line.startswith("Image"):
                if current_image and coordinates:
                    total_length = 0
                    for i in range(len(coordinates) - 1):
                        lat1, lon1 = coordinates[i]
                        lat2, lon2 = coordinates[i + 1]
                        total_length += euclidean_distance_km(lat1, lon1, lat2, lon2)
                    total_trajectory_lengths[current_image] = total_length
                current_image = line.rstrip(':')
                coordinates = []

            elif "Latitude" in line:
                lat = float(line.split(":")[1].strip())
                coordinates.append((lat, coordinates[-1][1] if coordinates else None))
            elif "Longitude" in line:
                lon = float(line.split(":")[1].strip())
                coordinates[-1] = (coordinates[-1][0], lon)

        if current_image and coordinates:
            total_length = 0
            for i in range(len(coordinates) - 1):
                lat1, lon1 = coordinates[i]
                lat2, lon2 = coordinates[i + 1]
                total_length += euclidean_distance_km(lat1, lon1, lat2, lon2)
            total_trajectory_lengths[current_image] = total_length

    return total_trajectory_lengths

def trajectory_results(mp_file, vlm_file, output_file="traj_length.txt", total_images=30):
    """Main function to calculate trajectory results and save them."""
    mp_lengths = calculate_trajectory(mp_file)
    vlm_lengths = calculate_trajectory(vlm_file)

    total_vlm_length = 0
    total_mp_length = 0

    with open(output_file, 'w') as file:
        for i in range(1, total_images + 1):
            image = f"Image {i}"
            mp_length = mp_lengths.get(image, 0)
            vlm_length = vlm_lengths.get(image, 0)

            # Total trajectory length for all the images from VLM and MP
            total_vlm_length += vlm_length
            total_mp_length += mp_length

            file.write(f"{image}:\n")
            file.write(f"Trajectory length from VLM: {vlm_length:.2f} km\n")
            file.write(f"Trajectory length from MP: {mp_length:.2f} km\n\n")

    print(f"Total trajectory length from VLM: {total_vlm_length:.2f} km")
    print(f"Total trajectory length from MP: {total_mp_length:.2f} km")

# Example usage (for testing, in the actual main script you will call this)
mp_file = "mp_coordinates.txt"
vlm_file = "VLM_coordinates.txt"
output_file = "traj_length.txt"

trajectory_results(mp_file, vlm_file, output_file)
