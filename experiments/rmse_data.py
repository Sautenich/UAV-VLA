# rmse_data.py
import numpy as np
from scipy.spatial import cKDTree
from scipy.interpolate import interp1d
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
import re

# Convert latitude and longitude differences to meters
def lat_lon_to_meters(lat, lon):
    lat_m = lat * 111320  # 1 degree latitude = 111,320 meters
    lon_m = lon * 111320 * np.cos(np.radians(lat))  # Adjust longitude by latitude
    return lat_m, lon_m

# Euclidean distance in meters between two lat/lon points
def euclidean_distance_meters(lat1, lon1, lat2, lon2):
    lat1_m, lon1_m = lat_lon_to_meters(lat1, lon1)
    lat2_m, lon2_m = lat_lon_to_meters(lat2, lon2)
    return np.sqrt((lat2_m - lat1_m)**2 + (lon2_m - lon1_m)**2)

def parse_coordinates(filename):
    coordinates = {}
    with open(filename, 'r') as file:
        data = file.read()
    
    images = re.split(r'Image \d+\:', data)
    for i, image_data in enumerate(images[1:], 1):
        image_name = f"Image {i}.jpg"
        coords = []
        matches = re.findall(r'Latitude: ([\d.-]+)\s+Longitude: ([\d.-]+)', image_data)
        
        for match in matches:
            lat, lon = float(match[0]), float(match[1])
            coords.append((lat, lon))
        
        coordinates[image_name] = coords
    
    return coordinates

def interpolate_path(path, num_points):
    lats, lons = zip(*path)
    x = np.linspace(0, 1, len(path))
    x_interp = np.linspace(0, 1, num_points)
    
    lat_interp = interp1d(x, lats, kind='linear')(x_interp)
    lon_interp = interp1d(x, lons, kind='linear')(x_interp)
    
    return list(zip(lat_interp, lon_interp))

# Compute nearest neighbor error (KNN)
def compute_nearest_neighbor_euclidean_error(trajectory1, trajectory2):
    traj1_meters = [lat_lon_to_meters(lat, lon) for lat, lon in trajectory1]
    traj2_meters = [lat_lon_to_meters(lat, lon) for lat, lon in trajectory2]
    tree1 = cKDTree(traj1_meters)
    tree2 = cKDTree(traj2_meters)
    distances1, _ = tree1.query(traj2_meters)
    distances2, _ = tree2.query(traj1_meters)
    avg_euclidean_error = (np.mean(distances1) + np.mean(distances2)) / 2
    return avg_euclidean_error

# Compute DTW MSE and RMSE
def compute_dtw_mse_rmse(trajectory1, trajectory2):
    traj1_meters = [lat_lon_to_meters(lat, lon) for lat, lon in trajectory1]
    traj2_meters = [lat_lon_to_meters(lat, lon) for lat, lon in trajectory2]
    distance, path = fastdtw(traj1_meters, traj2_meters, dist=euclidean)
    squared_errors = [np.linalg.norm(np.array(traj1_meters[i]) - np.array(traj2_meters[j]))**2 for i, j in path]
    mse = np.mean(squared_errors)
    rmse = np.sqrt(mse)
    return mse, rmse

def calculate_rmse(mp_file, VLM_file, output_file):
    mp_coordinates = parse_coordinates(mp_file)
    vlm_coordinates = parse_coordinates(VLM_file)

    with open(output_file, "w") as file:
        for image in mp_coordinates.keys():
            if image in vlm_coordinates:
                mp_path = mp_coordinates[image]
                vlm_path = vlm_coordinates[image]
                
                if mp_path and vlm_path:
                    knn_error = compute_nearest_neighbor_euclidean_error(mp_path, vlm_path)
                    mse_dtw, rmse_dtw = compute_dtw_mse_rmse(mp_path, vlm_path)
                    num_points = max(len(mp_path), len(vlm_path))
                    mp_path_interp = interpolate_path(mp_path, num_points)
                    vlm_path_interp = interpolate_path(vlm_path, num_points)
                    squared_errors_interp = [
                        euclidean_distance_meters(lat1, lon1, lat2, lon2)**2
                        for (lat1, lon1), (lat2, lon2) in zip(mp_path_interp, vlm_path_interp)
                    ]
                    mse_interp = np.mean(squared_errors_interp)
                    rmse_interp = np.sqrt(mse_interp)
                    
                    file.write(f"\nResults for {image}:\n")
                    file.write(f"Nearest Neighbor Error (KNN): {knn_error:.6f} meters\n")
                    file.write(f"DTW-based Root Mean Squared Error (RMSE): {rmse_dtw:.6f} meters\n")
                    file.write(f"Interpolation-based Root Mean Squared Error (RMSE): {rmse_interp:.6f} meters\n")
                else:
                    file.write(f"Error: Path for {image} is empty in one of the files.\n")
            else:
                file.write(f"Path for {image} not found in both files.\n")
