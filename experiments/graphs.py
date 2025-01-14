import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import pandas as pd
from scipy.interpolate import make_interp_spline

def read_trajectory_data(file_path):
    vlm_lengths = []
    mp_lengths = []
    images = []
    with open(file_path, "r") as file:
        for line in file:
            line = line.strip()
            if line.startswith("Image"):
                images.append(int(line.split()[1].replace(":", ""))) 
            elif "Trajectory length from VLM" in line:
                vlm_lengths.append(float(line.split(":")[1].strip().split()[0]))
            elif "Trajectory length from MP" in line:
                mp_lengths.append(float(line.split(":")[1].strip().split()[0]))
    return images, vlm_lengths, mp_lengths

def plot_trajectory_length_comparison(vlm_lengths, mp_lengths, deviations, images, file_path="results/traj_bar_chart.png"):
    x = np.arange(len(images))
    x_smooth = np.linspace(x.min(), x.max(), 500)
    deviation_smooth = make_interp_spline(x, deviations)(x_smooth)

    matplotlib.rcParams.update({'font.size': 14})

    plt.figure(figsize=(12, 6))
    plt.bar(x - 0.2, vlm_lengths, width=0.4, label='VLM', align='center', alpha=0.8, color='#8A2BE2')
    plt.bar(x + 0.2, mp_lengths, width=0.4, label='MP', align='center', alpha=0.8, color='#FFD700')
    plt.plot(x_smooth, deviation_smooth, label='Deviation', color='black', linewidth=2)
    plt.xticks(x, images, fontsize=12, rotation=0)  
    plt.xlabel("Images", fontsize=16)  
    plt.ylabel("Trajectory Length (km)", fontsize=16) 
    plt.title("Comparison of Trajectory Lengths from VLM and MP with Deviations", fontsize=18)  
    plt.legend(fontsize=14)  
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(file_path)  
    plt.close()

def read_rmse_data(file_path):
    images = []
    knn_errors = []
    dtw_rmse = []
    interp_rmse = []

    with open(file_path, "r") as file:
        for line in file:
            line = line.strip()
            if line.startswith("Results for Image"):
                images.append(line.split(":")[0].split()[-1])
            elif "Nearest Neighbor Error (KNN)" in line:
                knn_errors.append(float(line.split(":")[1].strip().split()[0]))
            elif "DTW-based Root Mean Squared Error (RMSE)" in line:
                dtw_rmse.append(float(line.split(":")[1].strip().split()[0]))
            elif "Interpolation-based Root Mean Squared Error (RMSE)" in line:
                interp_rmse.append(float(line.split(":")[1].strip().split()[0]))
    
    return images, knn_errors, dtw_rmse, interp_rmse

def plot_rmse_boxplot(knn_errors, dtw_rmse, interp_rmse, file_path="results/error_box_plot.png"):
    data_dict = {
        "KNN Error": knn_errors,
        "DTW RMSE": dtw_rmse,
        "Interpolation RMSE": interp_rmse
    }
    plt.figure(figsize=(8, 6))
    box = plt.boxplot(data_dict.values(), labels=data_dict.keys(), patch_artist=True, showmeans=False)

    for patch in box['boxes']:
        patch.set_facecolor('#8A2BE2')  

    for median_line in box['medians']:
        median_line.set_color('#FFD700')
        median_line.set_linewidth(2)

    plt.yscale('log')  # Apply logarithmic scale to y-axis
    plt.title("Error Distribution b/w VLM and MP Methods (Log Scale)", fontsize=18)  
    plt.ylabel("Error (meters)", fontsize=16) 
    plt.xticks(fontsize=12) 
    plt.yticks(fontsize=12) 
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(file_path)  
    plt.close()

def print_rmse_comparison_table(knn_errors, dtw_rmse, interp_rmse):
    from tabulate import tabulate
    aggregates = {
        "Metric": ["Mean", "Median", "Max"],
        "KNN Error": [np.mean(knn_errors), np.median(knn_errors), np.max(knn_errors)],
        "DTW RMSE": [np.mean(dtw_rmse), np.median(dtw_rmse), np.max(dtw_rmse)],
        "Interpolation RMSE": [np.mean(interp_rmse), np.median(interp_rmse), np.max(interp_rmse)],
    }
    print(tabulate(pd.DataFrame(aggregates), headers="keys", tablefmt="grid"))
