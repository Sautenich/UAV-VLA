import os
import home_pose
import VLM_data
import mp_data
import traj_calc
import rmse_data
import graphs
import identified_images  

def main():
    # Make sure the output folders exist
    os.makedirs('results', exist_ok=True)
    os.makedirs('identified_images_VLM', exist_ok=True)
    os.makedirs('../benchmark-UAV-VLPA-nano-30/identified_images_mp', exist_ok=True)

    # Step 1: Generate Home Positions
    print("Generating home positions...")
    home_pose_file = 'home_position.txt'
    home_pose.process_image_data('../benchmark-UAV-VLPA-nano-30/img_lat_long_data.txt', home_pose_file)

    # Step 2: Generate VLM Coordinates
    print("Generating VLM coordinates...")
    VLM_data_file = 'VLM_coordinates.txt'
    VLM_data.process_vlm_data(
        latlong_file='../benchmark-UAV-VLPA-nano-30/img_lat_long_data.txt', 
        stadium_file='../identified_points.txt', 
        home_position_file='home_position.txt', 
        output_file=VLM_data_file
    )

    # Step 3: Generate MP Coordinates
    print("Generating MP coordinates...")
    mp_file = 'mp_coordinates.txt'
    mp_data.process_mp_data(
        input_directory='../benchmark-UAV-VLPA-nano-30/mission_planner_data/waypoints/',
        output_txt=mp_file
    )

    # Step 4: Calculate Trajectory Lengths
    print("Calculating trajectory lengths...")
    traj_calc.trajectory_results(mp_file, VLM_data_file)

    # Step 5: Calculate RMSE Data
    print("Calculating RMSE...")
    rmse_data_file = 'rmse.txt'
    rmse_data.calculate_rmse(mp_file, VLM_data_file, rmse_data_file)

    # Step 6: Plot Results
    print("Plotting results...")

    # Plot Trajectory Length Comparison
    images, vlm_lengths, mp_lengths = graphs.read_trajectory_data("traj_length.txt")
    deviations = [vlm - mp for vlm, mp in zip(vlm_lengths, mp_lengths)]
    graphs.plot_trajectory_length_comparison(vlm_lengths, mp_lengths, deviations, images)

    # Plot RMSE Boxplot
    images, knn_errors, dtw_rmse, interp_rmse = graphs.read_rmse_data(rmse_data_file)
    graphs.plot_rmse_boxplot(knn_errors, dtw_rmse, interp_rmse)

    # Print RMSE Comparison Table
    graphs.print_rmse_comparison_table(knn_errors, dtw_rmse, interp_rmse)

    # Step 7: Generate Identified Images for VLM
    print("Generating identified images for VLM...")
    identified_images.process_images(
        input_folder='../benchmark-UAV-VLPA-nano-30/images/',  # Adjust this path as needed
        output_folder='identified_images_VLM',  # VLM output folder
        coordinate_file='VLM_coordinates.txt', 
        latlong_file='../benchmark-UAV-VLPA-nano-30/img_lat_long_data.txt', 
        is_vlm=True
    )

    # Step 8: Generate Identified Images for MP
    print("Generating identified images for MP...")
    identified_images.process_images(
        input_folder='../benchmark-UAV-VLPA-nano-30/images/',  # Adjust this path as needed
        output_folder='../benchmark-UAV-VLPA-nano-30/identified_images_mp',  # MP output folder
        coordinate_file='mp_coordinates.txt', 
        latlong_file='../benchmark-UAV-VLPA-nano-30/img_lat_long_data.txt', 
        is_vlm=False
    )

if __name__ == "__main__":
    main()
