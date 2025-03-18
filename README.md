# UAV-VLA: Vision-Language-Action System for Large Scale Aerial Mission Generation

##### UPD from the authors: We are a bit surprised about the popularity of this paper, so the code and data are about to be refactored for more convenient format. 
##### Oleg 

---
[![arXiv](https://img.shields.io/badge/https://arxiv.org/abs/2501.05014-df2a2a.svg?style=for-the-badge)](https://arxiv.org/abs/2501.05014)
https://img.shields.io/badge/https%3A%2F%2Farxiv.org%2Fabs%2F2503.02454-Paper



This repository is for the research paper accepted in Proc. ACM/IEEE Int. Conf. on Human Robot Interaction (HRI 2025)

## Table of Contents
1. [Abstract](#abstract)
2. [Benchmark](#benchmark)
3. [Installation](#installation)
4. [Mission Generation](#mission-generation)
5. [Path-Plans Creation](#path-plan-creation)
6. [Experimental Results](#experimental-results)
7. [Simulation Video](simulation-video)
8. [Citation](citation)


---
## Abstract
The UAV-VLPA* (Visual-Language-Planning-and-Action) system represents a cutting-edge advancement in aerial robotics, designed to enhance communication and operational efficiency for unmanned aerial vehicles (UAVs). By integrating advanced planning capabilities, the system addresses the Traveling Salesman Problem (TSP) to optimize flight paths, reducing the total trajectory length by 18.5% compared to traditional methods. Additionally, the incorporation of the A* algorithm enables robust obstacle avoidance, ensuring safe and efficient navigation in complex environments. The system leverages satellite imagery processing combined with the Visual Language Model (VLM) and GPT’s natural language processing capabilities, allowing users to generate detailed flight plans through simple text commands.
This seamless fusion of visual and linguistic analysis empowers precise decision-making and mission planning, making UAV-VLPA* a transformative tool for modern aerial operations. With its unmatched operational efficiency, navigational safety, and user-friendly functionality, UAV-VLPA* sets a new standard in autonomous aerial robotics, paving the way for future innovations in the field.



This repository includes:
- The implementation of the UAV-VLPA framework.

### UAV-VLPA Framework

<div align="center">
  <img src="![image](https://github.com/user-attachments/assets/2a8a3560-9a66-4721-a5aa-be5dbb8a8940)" alt="UAV_VLPA_Title_image" width="400"/>
</div>

## Benchmark

The images of the benchmark are stored in the folder ```benchmark-UAV-VLPA-nano-30/images```. The metadata files are ```benchmark-UAV-VLPA-nano-30/img_lat_long_data.txt``` and ```benchmark-UAV-VLPA-nano-30/parsed_coordinates.csv```.

## Installation


To install requirements, run 

```
pip -r requirements.txt
```
!12GB VRAM minimum

## Export your ChatGpt api key
```
export api_key="your chatgpt ap_key"
```

## Mission generation

To generate commands for UAV add your API key for ChatGPT in the generate_plans.py, then run
```
python3 generate_plans.py
```
It will produce the commands and store the text files in the folder ```/created_missions``` and visualizations of the identified points on the benchmark images in the folder ```/identified_new_data```.

As a result of this script, you will also find the total computational time time of the UAV-VLA system which is approximately **5 minutes and 24 seconds**.

## Path-Plans Creation

To see the results of VLM on the benchmark, run
```
python3 run_vlm.py
```

Some examples of the path generated can be seen below:

<div align="center">
  <img src="https://github.com/user-attachments/assets/aa7595f3-9961-439f-8e05-36877cfee6cb" alt="examples_path_generated" width="600"/>
</div>

## Experimental Results

To view the experimental results, you need to run the main.py script. This script automates the entire process of generating coordinates, calculating trajectory lengths, and producing visualizations.

Navigate into the folder ```experiments/```, run:
```
python3 main.py
```

### What Happens When You run main.py:

- Generate Home Positions

- Generate VLM Coordinates

- Generate MP Coordinates

- Calculate Trajectory Lengths

- Calculate RMSE (Root Mean Square Error)
  
- Plot Results

- Generate Identified Images:
The script generates images by overlaying the VLM and Mission Planner (human-generated) coordinates on the original images from the dataset.
These identified images are saved in ```identified_images_VLM/``` (for VLM outputs) and ```identified_images_mp/``` (for Mission Planner outputs).

After running the script, you will be able to examine:

- Text Files: Containing the generated coordinates, home positions, and RMSE data.
- Images: Showing the identified coordinates overlaid on the images.
- Plots: Comparing trajectory lengths and RMSE values.

### Trajectory Bar Chart:

<div align="center">
  <img src="https://github.com/user-attachments/assets/e27a0c86-e54a-433a-822c-dc68297fdd37" alt="traj_bar_chart" width="600"/>
</div>

### Error Box Plot:

<div align="center">
  <img src="https://github.com/user-attachments/assets/52f9afcf-ba3f-4cc2-bb37-bf48475a077b" alt="error_box_plot" width="500"/>
</div>

### Error Comparison Table:

The errors were calculated using different approaches including K-Nearest Neighbor (KNN), Dynamic Time Warping (DTW), and Linear Interpolation.

<div align="center">

|    | Metric   |   KNN Error (m) |   DTW RMSE (m) |   Interpolation RMSE (m) |
|----|----------|-----------------|----------------|--------------------------|
|  1 | Mean     |     34.2218     |    307.265     |              409.538     |
|  2 | Median   |     26.0456     |    318.462     |              395.593     |
|  3 | Max      |    112.493      |    644.574     |              727.936     |

</div>

## Simulation Video

The generated mission from the UAV-VLA framework was tested in the ArduPilot Mission Planner. The simulation can be seen below.

https://github.com/user-attachments/assets/562f2ee7-13e5-44a0-bb0f-6c109a958123

## Citation
``` bash
@inproceedings{10.5555/3721488.3721725,
author = {Sautenkov, Oleg and Yaqoot, Yasheerah and Lykov, Artem and Mustafa, Muhammad Ahsan and Tadevosyan, Grik and Akhmetkazy, Aibek and Altamirano Cabrera, Miguel and Martynov, Mikhail and Karaf, Sausar and Tsetserukou, Dzmitry},
title = {UAV-VLA: Vision-Language-Action System for Large Scale Aerial Mission Generation},
year = {2025},
publisher = {IEEE Press},
abstract = {The UAV-VLA (Visual-Language-Action) system is a tool designed to facilitate communication with aerial robots. By integrating satellite imagery processing with the Visual Language Model (VLM) and the powerful capabilities of GPT, UAV-VLA enables users to generate general flight paths-and-action plans through simple text requests. This system leverages the rich contextual information provided by satellite images, allowing for enhanced decision-making and mission planning. The combination of visual analysis by VLM and natural language processing by GPT can provide the user with the path-and-action set, making aerial operations more efficient and accessible. The newly developed method showed the difference in the length of the created trajectory in 22\% and the mean error in finding the objects of interest on a map in 34.22 m by Euclidean distance in the K-Nearest Neighbors (KNN) approach. Additionally, the UAV-VLA system generates all flight plans in just 5 minutes and 24 seconds, making it 6.5 times faster than an experienced human operator.},
booktitle = {Proceedings of the 2025 ACM/IEEE International Conference on Human-Robot Interaction},
pages = {1588–1592},
numpages = {5},
keywords = {drone, llm-agents, navigation, path planning, uav, vla, vlm, vlm-agents},
location = {Melbourne, Australia},
series = {HRI '25}
}



