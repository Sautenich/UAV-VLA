# UAV-VLA: Vision-Language-Action System for Large Scale Aerial Mission Generation


---
[![arXiv](https://img.shields.io/badge/https://arxiv.org/abs/2501.05014-df2a2a.svg?style=for-the-badge)](https://arxiv.org/abs/2501.05014)

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
The UAV-VLA (Visual-Language-Action) system is a tool designed to facilitate communication with aerial robots. 
By integrating satellite imagery processing with Gemini multimodal models, UAV-VLA enables users to generate general flight paths-and-action plans through simple text requests. 
This system leverages the rich contextual information provided by satellite images, allowing for enhanced decision-making and mission planning. 
The current implementation uses Gemini both for visual object localization on satellite images and for natural-language mission generation. The original benchmark results showed the difference in the length of the created trajectory in 22\% and the mean error in finding the objects of interest on a map in 34.22 m by Euclidean distance in the K-Nearest Neighbors (KNN) approach.

https://arxiv.org/abs/2501.05014



This repository includes:
- The implementation of the UAV-VLA framework.
- Dataset and benchmark details.
- Archived code and artifacts for simulation-based experiments in Mission Planner.

### UAV-VLA Framework

<div align="center">
  <img src="https://raw.githubusercontent.com/Sautenich/UAV-VLA/refs/heads/refactored/uavplanner.png" alt="UAV_VLA_Title_image" width="400"/>
</div>

## Benchmark

The images of the benchmark are stored in the folder ```benchmark-UAV-VLPA-nano-30/images```. The metadata file used by the current pipeline is ```benchmark-UAV-VLPA-nano-30/parsed_coordinates.csv```, which stores image corner coordinates for converting image-percentage points to latitude/longitude.

Additional historical Mission Planner and experiment files are stored in ```archive/```.

## Installation


To install requirements, run 

```
pip install -r requirements.txt
```

The current Gemini-based pipeline runs through an API and does not require a local GPU. The archived Molmo-based legacy pipeline required a CUDA GPU with substantial VRAM.

## Export your Gemini API key
```
export GEMINI_API_KEY="your gemini api_key"
```

Alternatively, create a local ```.env``` file in the repository root:
```
GEMINI_API_KEY="your gemini api_key"
GEMINI_MODEL="gemini-2.5-flash"
NUMBER_OF_SAMPLES=30
GEMINI_MAX_RETRIES=5
GEMINI_RETRY_DELAY_SECONDS=10
GEMINI_SAMPLE_DELAY_SECONDS=5
```

## Mission generation

To generate commands for UAV, add your Gemini API key to the environment or to a local ```.env``` file, then run:
```
python3 generate_plans.py
```

The script will:
- extract building-like object types from the mission command;
- send benchmark satellite images to Gemini Vision for object localization;
- convert Gemini image-percentage coordinates to latitude/longitude;
- save annotated images to ```identified_new_data/```;
- generate mission command files in ```created_missions/```.

To run the full pipeline for only one image, pass the image filename:
```
python3 generate_plans.py 1.jpg
```

## Path-Plans Creation

The current path-plan creation is part of ```generate_plans.py```. It produces pseudo-language UAV mission files in ```created_missions/```, for example:
```
arm throttle
takeoff 100
mode guided 42.87946931666667 -85.4114592638889 100
mode circle
mode rtl
disarm
```

The old local Molmo VLM runner has been moved to ```archive/legacy_molmo/```.

Some examples of the path generated can be seen below:

<div align="center">
  <img src="https://github.com/user-attachments/assets/386f7e78-83aa-4915-aa33-ec7fbdc6dd40" alt="examples_path_generated" width="600"/>
</div>

## Experimental Results

The original experimental pipeline and generated artifacts are stored in ```archive/experiments/```. This code was used to compare VLM-generated coordinates with Mission Planner data, calculate trajectory lengths, calculate RMSE metrics, and produce visualizations.

If you want to inspect or rerun the archived experiment code, navigate into the folder ```archive/experiments/```, then run:
```
python3 main.py
```

Note: the archived experiment runner expects the historical VLM output file ```identified_points.txt```. The current Gemini pipeline does not generate that file by default; it writes annotated images to ```identified_new_data/``` and missions to ```created_missions/```.

### What Happens When You run main.py:

- Generate Home Positions

- Generate VLM Coordinates

- Generate MP Coordinates

- Calculate Trajectory Lengths

- Calculate RMSE (Root Mean Square Error)
  
- Plot Results

- Generate Identified Images:
The script generates images by overlaying the VLM and Mission Planner (human-generated) coordinates on the original images from the dataset.
These identified images are saved in ```archive/experiments/identified_images_VLM/``` (for VLM outputs) and ```archive/identified_images_mp/``` (for Mission Planner outputs).

After running the script, you will be able to examine:

- Text Files: Containing the generated coordinates, home positions, and RMSE data.
- Images: Showing the identified coordinates overlaid on the images.
- Plots: Comparing trajectory lengths and RMSE values.

### Trajectory Bar Chart:

<div align="center">
  <img width="800"alt="traj_bar_chart" src="https://github.com/user-attachments/assets/5a701ef3-5cfc-4b47-94d6-7866a14077a8" />
</div>

### Error Box Plot:

<div align="center">
  <img width="500"alt="error_boxplot" src="https://github.com/user-attachments/assets/619ac3d2-c354-4925-8036-b89e69cba34e" />
</div>

### Error Comparison Table:

The errors were calculated using different approaches including K-Nearest Neighbor (KNN), Dynamic Time Warping (DTW), and Linear Interpolation.

<div align="center">

|   # | Metric | KNN Error (m) | DTW RMSE (m) | Interpolation RMSE (m) |
|-----|--------|---------------|--------------|------------------------|
|   1 | Mean   |     45.91     |    291.94    |           403.03       |
|   2 | Median |     30.26     |    279.58    |           376.55       |
|   3 | Max    |    299.04     |    711.40    |           738.07       |

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
```
