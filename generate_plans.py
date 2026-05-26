"""
UAV Mission Generation Module

This module handles the generation of UAV missions using a combination of
Vision-Language Models (VLM) and Large Language Models (LLM).
"""

import json
import os
import base64
import mimetypes
import sys
from pathlib import Path
from typing import Tuple, List, Dict, Any, Optional
import logging
import requests
from time import sleep, time

from draw_circles import draw_dots_and_lines_on_image
from recalculate_to_latlon import recalculate_coordinates, read_coordinates_from_csv
from config import *

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_env_file(file_path: str = ".env") -> None:
    """Load simple KEY=VALUE pairs from a local .env file into os.environ."""
    if not os.path.exists(file_path):
        return

    with open(file_path, "r") as env_file:
        for raw_line in env_file:
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue

            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")

            if key and key not in os.environ:
                os.environ[key] = value

def get_gemini_api_key() -> str:
    """Read a Gemini API key from the environment."""
    api_key = (
        os.environ.get("GEMINI_API_KEY")
        or os.environ.get("GOOGLE_API_KEY")
        or os.environ.get("api_key")
    )
    if not api_key:
        raise ValueError("Gemini API key not found in environment variables")
    return api_key

def call_gemini_parts(parts: List[Dict[str, Any]], api_key: str, model_name: str, temperature: float = 0.0) -> str:
    """Generate text with Gemini through the REST generateContent endpoint."""
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent"
    payload = {
        "contents": [
            {
                "role": "user",
                "parts": parts
            }
        ],
        "generationConfig": {
            "temperature": temperature
        }
    }

    max_retries = get_int_env("GEMINI_MAX_RETRIES", GEMINI_MAX_RETRIES)
    retry_delay = get_float_env("GEMINI_RETRY_DELAY_SECONDS", GEMINI_RETRY_DELAY_SECONDS)
    response = None

    for attempt in range(1, max_retries + 1):
        response = requests.post(
            url,
            params={"key": api_key},
            json=payload,
            timeout=60
        )

        if response.ok:
            break

        try:
            error_message = response.json().get("error", {}).get("message", response.text)
        except ValueError:
            error_message = response.text

        if response.status_code not in (429, 503) or attempt == max_retries:
            raise RuntimeError(f"Gemini API request failed: {response.status_code} {error_message}")

        delay = retry_delay * attempt
        logger.warning(
            f"Gemini API request failed with {response.status_code}: {error_message}. "
            f"Retrying in {delay:.1f}s ({attempt}/{max_retries})"
        )
        sleep(delay)

    data = response.json()
    candidates = data.get("candidates", [])
    if not candidates:
        raise RuntimeError("Gemini API returned no candidates")

    parts = candidates[0].get("content", {}).get("parts", [])
    text_parts = [part.get("text", "") for part in parts if part.get("text")]
    if not text_parts:
        raise RuntimeError("Gemini API returned no text")

    return "".join(text_parts).strip()

def call_gemini(prompt: str, api_key: str, model_name: str, temperature: float = 0.0) -> str:
    """Generate text with Gemini."""
    return call_gemini_parts([{"text": prompt}], api_key, model_name, temperature)

def get_int_env(name: str, default: int) -> int:
    """Read an integer setting from environment variables."""
    value = os.environ.get(name)
    if value is None:
        return default

    try:
        return int(value)
    except ValueError:
        logger.warning(f"Invalid {name} value: {value}. Using default: {default}")
        return default

def get_float_env(name: str, default: float) -> float:
    """Read a float setting from environment variables."""
    value = os.environ.get(name)
    if value is None:
        return default

    try:
        return float(value)
    except ValueError:
        logger.warning(f"Invalid {name} value: {value}. Using default: {default}")
        return default

def strip_json_markdown(text: str) -> str:
    """Remove common markdown fences around JSON model output."""
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`").strip()
        if cleaned.lower().startswith("json"):
            cleaned = cleaned[4:].strip()
    return cleaned

def parse_json_response(text: str) -> Dict[str, Any]:
    """Parse a Gemini JSON response with a small fallback for surrounding prose."""
    cleaned = strip_json_markdown(text)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise
        return json.loads(cleaned[start:end + 1])

def encode_image_part(image_path: str) -> Dict[str, Any]:
    """Encode an image as a Gemini inline_data part."""
    mime_type = mimetypes.guess_type(image_path)[0] or "image/jpeg"
    with open(image_path, "rb") as image_file:
        image_data = base64.b64encode(image_file.read()).decode("utf-8")

    return {
        "inline_data": {
            "mime_type": mime_type,
            "data": image_data
        }
    }

def detect_objects_with_gemini(
    image_path: str,
    object_types: List[str],
    api_key: str,
    model_name: str
) -> Dict[str, Dict[str, Any]]:
    """Identify requested objects in an image using Gemini Vision."""
    prompt = object_detection_template.format(object_types=", ".join(object_types))
    response_text = call_gemini_parts(
        [
            encode_image_part(image_path),
            {"text": prompt}
        ],
        api_key,
        model_name
    )
    return parse_json_response(response_text)

def resolve_image_path(image_filename: str) -> Tuple[str, int, str]:
    """Resolve an image filename/path and return path, numeric id, and file stem."""
    image_path = Path(image_filename)
    if not image_path.exists():
        image_path = Path(IMAGES_DIR) / image_filename

    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_filename}")

    try:
        image_number = int(image_path.stem)
    except ValueError as exc:
        raise ValueError(f"Image filename must have a numeric stem, got: {image_path.name}") from exc

    return str(image_path), image_number, image_path.stem

def process_single_image(
    image_filename: str,
    object_types: List[str],
    api_key: str,
    model_name: str,
    coordinates_dict: Optional[Dict[str, Tuple[float, float, float, float]]] = None,
    mission_command: str = command,
    generate_mission: bool = True
) -> Dict[str, Any]:
    """Run object detection, coordinate conversion, visualization, and mission generation for one image."""
    image_path, image_number, image_stem = resolve_image_path(image_filename)
    if coordinates_dict is None:
        coordinates_dict = read_coordinates_from_csv(COORDINATES_FILE)

    parsed_points = detect_objects_with_gemini(
        image_path,
        object_types,
        api_key,
        model_name
    )
    logger.debug(f'Parsed points for image {image_stem}: {parsed_points}')

    result_coordinates = recalculate_coordinates(parsed_points, image_number, coordinates_dict)

    output_path = f'{IDENTIFIED_DATA_DIR}/identified{image_stem}.jpg'
    draw_dots_and_lines_on_image(image_path, parsed_points, output_path=output_path)

    flight_plan_text = ""
    mission_file = None
    if generate_mission:
        os.makedirs(MISSION_OUTPUT_DIR, exist_ok=True)
        mission_prompt = step_3_template.format(
            command=mission_command,
            objects=json.dumps(result_coordinates)
        )
        flight_plan_text = call_gemini(mission_prompt, api_key, model_name)
        mission_file = f"{MISSION_OUTPUT_DIR}/mission{image_stem}.txt"
        with open(mission_file, "w") as file:
            file.write(flight_plan_text)

    return {
        "image_path": image_path,
        "annotated_image_path": output_path,
        "mission_file": mission_file,
        "percentage_coordinates": parsed_points,
        "latlon_coordinates": result_coordinates,
        "flight_plan": flight_plan_text
    }

def generate_drone_mission_for_image(image_filename: str, mission_command: str = command) -> Dict[str, Any]:
    """Run the full UAV-VLA pipeline for a single image filename/path."""
    load_env_file()
    api_key = get_gemini_api_key()
    model_name = os.environ.get("GEMINI_MODEL", GEMINI_MODEL)

    object_types_prompt = step_1_template.format(command=mission_command)
    object_types_json = call_gemini(object_types_prompt, api_key, model_name)
    object_types = parse_json_response(object_types_json)["object_types"]

    return process_single_image(
        image_filename=image_filename,
        object_types=object_types,
        api_key=api_key,
        model_name=model_name,
        mission_command=mission_command
    )

def find_objects(
    json_input: str,
    example_objects: str,
    api_key: str,
    model_name: str
) -> Tuple[str, List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Identify objects in satellite images using Gemini Vision.
    
    Args:
        json_input: JSON string containing object types to search for
        example_objects: Example object format for reference
        api_key: Gemini API key
        model_name: Gemini model name
        
    Returns:
        Tuple containing:
        - JSON string of result coordinates
        - List of percentage coordinates
        - List of lat/lon coordinates
    """
    list_of_the_resulted_coordinates_percentage = []
    list_of_the_resulted_coordinates_lat_lon = []
    last_result_coordinates = {}
    
    try:
        find_objects_json_input = parse_json_response(json_input)
        object_types = find_objects_json_input["object_types"]

        sample_count = get_int_env("NUMBER_OF_SAMPLES", NUMBER_OF_SAMPLES)
        sample_delay = get_float_env("GEMINI_SAMPLE_DELAY_SECONDS", GEMINI_SAMPLE_DELAY_SECONDS)
        csv_file_path = 'benchmark-UAV-VLPA-nano-30/parsed_coordinates.csv'
        coordinates_dict = read_coordinates_from_csv(csv_file_path)
            
        logger.info(f'Processing {sample_count} samples')
        
        for i in range(1, sample_count+1):
            logger.info(f'Processing image {i}')
            
            try:
                image_result = process_single_image(
                    image_filename=f"{i}.jpg",
                    object_types=object_types,
                    api_key=api_key,
                    model_name=model_name,
                    coordinates_dict=coordinates_dict,
                    mission_command=command,
                    generate_mission=False
                )
                parsed_points = image_result["percentage_coordinates"]
                result_coordinates = image_result["latlon_coordinates"]
                last_result_coordinates = result_coordinates

                list_of_the_resulted_coordinates_percentage.append(parsed_points)
                list_of_the_resulted_coordinates_lat_lon.append(result_coordinates)

                if sample_delay > 0 and i < sample_count:
                    logger.info(f'Waiting {sample_delay:.1f}s before next Gemini image request')
                    sleep(sample_delay)
                
            except Exception as e:
                logger.error(f'Error processing image {i}: {str(e)}')
                continue
                
    except Exception as e:
        logger.error(f'Error in find_objects: {str(e)}')
        raise
        
    return json.dumps(last_result_coordinates), list_of_the_resulted_coordinates_percentage, list_of_the_resulted_coordinates_lat_lon

def generate_drone_mission(command: str) -> Tuple[str, float, float]:
    """
    Generate a complete drone mission plan.
    
    Args:
        command: Natural language command describing the mission
        
    Returns:
        Tuple containing:
        - Flight plan text
        - Time taken to find objects
        - Time taken to generate mission
    """
    try:
        load_env_file()
        api_key = get_gemini_api_key()
        model_name = os.environ.get("GEMINI_MODEL", GEMINI_MODEL)
        

        # Step 1: Extract object types
        object_types_prompt = step_1_template.format(command=command)
        object_types_json = call_gemini(object_types_prompt, api_key, model_name)
        
        # Step 2: Find objects on the map
        t1_find_objects = time()
        objects_json, coords_percentage, coords_latlon = find_objects(
            object_types_json,
            example_objects,
            api_key,
            model_name
        )
        t2_find_objects = time()
        del_t_find_objects = (t2_find_objects - t1_find_objects)/60
        
        logger.info(f'Found {len(coords_latlon)} coordinate sets')
        
        # Step 3: Generate flight plans
        t1_generate_drone_mission = time()
        os.makedirs("created_missions", exist_ok=True)
        flight_plan_text = ""

        for i, coords in enumerate(coords_latlon, 1):
            mission_prompt = step_3_template.format(
                command=command,
                objects=json.dumps(coords)
            )
            flight_plan_text = call_gemini(mission_prompt, api_key, model_name)

            mission_file = f"created_missions/mission{i}.txt"
            with open(mission_file, "w") as file:
                file.write(flight_plan_text)

            logger.info(f'Generated mission plan {i}')

        t2_generate_drone_mission = time()
        del_t_generate_drone_mission = (t2_generate_drone_mission - t1_generate_drone_mission)/60
        
        return flight_plan_text, del_t_find_objects, del_t_generate_drone_mission
        
    except Exception as e:
        logger.error(f'Error in generate_drone_mission: {str(e)}')
        raise

def run():
    """Main entry point for the UAV mission generation system."""
    try:
        load_env_file()
        sample_count = get_int_env("NUMBER_OF_SAMPLES", NUMBER_OF_SAMPLES)
        logger.info('Starting UAV mission generation')
        logger.info(f'Processing {sample_count} samples')
        
        flight_plan, vlm_time, mission_time = generate_drone_mission(command)
        total_time = vlm_time + mission_time
        
        logger.info('Mission generation complete')
        logger.info(f'VLM processing time: {vlm_time:.2f} mins')
        logger.info(f'Mission generation time: {mission_time:.2f} mins')
        logger.info(f'Total computational time: {total_time:.2f} mins')
        
    except Exception as e:
        logger.error(f'Error in main execution: {str(e)}')
        raise

if __name__ == "__main__":
    if len(sys.argv) > 1:
        result = generate_drone_mission_for_image(sys.argv[1])
        logger.info(f"Single-image pipeline complete. Mission saved to: {result['mission_file']}")
        logger.info(f"Annotated image saved to: {result['annotated_image_path']}")
    else:
        run()
