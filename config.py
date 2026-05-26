"""
Configuration Module for UAV-VLA System

This module contains all configuration parameters and templates used throughout
the UAV-VLA (Vision-Language-Action) system.
"""

from typing import Dict, Any

# System Configuration
NUMBER_OF_SAMPLES: int = 30
GEMINI_MODEL: str = "gemini-2.5-flash"
GEMINI_MAX_RETRIES: int = 5
GEMINI_RETRY_DELAY_SECONDS: int = 10
GEMINI_SAMPLE_DELAY_SECONDS: int = 5

# Example Data Structures
EXAMPLE_BUILDINGS: Dict[str, Dict[str, Any]] = {
    'building_1': {'type': 'building', 'coordinates': [40.2, 39.5]},
    'building_2': {'type': 'building', 'coordinates': [47.7, 39.0]},
    'building_3': {'type': 'building', 'coordinates': [64.9, 41.2]},
    'building_4': {'type': 'building', 'coordinates': [65.2, 87.9]},
    'building_5': {'type': 'building', 'coordinates': [80.2, 20.7]}
}

example_objects = '''
{
    "village_1": {"type": "village", "coordinates": [1.5, 3.5]},
    "village_2": {"type": "village", "coordinates": [2.5, 6.0]},
    "airfield": {"type": "airfield", "coordinates": [8.0, 6.5]}
}
'''

# Prompt Templates
# Previous generic object extraction prompt:
# step_1_template = """
# Extract all types of objects the drone needs to find from the following mission description:
# "{command}"
#
# Output the result in JSON format with a list of object types.
# Return only valid JSON. Do not include markdown fences or extra text.
# Example output:
# {{
#     "object_types": ["village", "airfield", "stadium", "tennis court", "building", "ponds", "crossroad", "roundabout"]
# }}
# """

step_1_template = """
Analyze the mission description and determine which man-made building-like structures the drone should visit:
"{command}"

The route must be built only through objects that visually look like buildings or constructed structures.
Ignore natural objects and landscape features such as trees, grass, fields, forests, rivers, ponds, lakes, bare ground, roads, parking lots, crossroads, roundabouts, and other non-building areas.

Return only valid JSON. Do not include markdown fences or extra text.
Use only building-focused object types in the output. If the mission mentions vague objects, map them to building-like structures only.
Example output:
{{
    "object_types": ["building", "house", "warehouse", "hangar", "industrial building", "large roofed structure"]
}}
"""

step_3_template = """
Given the mission description: "{command}" and the following identified objects: {objects}, generate a flight plan in pseudo-language.

Available commands:
- arm throttle: arm the copter
- takeoff Z: lift Z meters
- disarm: disarm the copter
- mode rtl: return to home
- mode circle: circle and observe at the current position
- mode guided(X Y Z): fly to the specified location

Example output:
arm throttle
mode guided 43.237763722222226 -85.79224314444444 100
mode guided 43.237765234234234 -85.79224314235235 100
mode circle
mode rtl
disarm
"""

object_detection_template = """
This is a satellite image.
Find all visible objects matching these requested object types: {object_types}.

Return only valid JSON. Do not include markdown fences or extra text.
Use percentage coordinates from 0 to 100, where x=0 is the left edge, x=100 is the right edge,
y=0 is the top edge, and y=100 is the bottom edge.

Output format:
{{
    "object_1": {{"type": "building", "coordinates": [40.2, 39.5]}},
    "object_2": {{"type": "building", "coordinates": [47.7, 39.0]}}
}}
"""

# Default mission command
command = "Create a flight plan for the quadcopter to fly around each of the building at the height 100m return to home and land at the take-off point."

# File paths
BENCHMARK_DIR = "benchmark-UAV-VLPA-nano-30"
IMAGES_DIR = f"{BENCHMARK_DIR}/images"
COORDINATES_FILE = f"{BENCHMARK_DIR}/parsed_coordinates.csv"
MISSION_OUTPUT_DIR = "created_missions"
IDENTIFIED_DATA_DIR = "identified_new_data"
