"""
UAV Mission Generation Module

This module handles the generation of UAV missions using a combination of
Vision-Language Models (VLM) and Large Language Models (LLM).
"""

from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
import json
from transformers import AutoModelForCausalLM, AutoProcessor, GenerationConfig
from PIL import Image
import torch
import os
from typing import Tuple, List, Dict, Any
import logging
from time import time

from parser_for_coordinates import parse_points
from draw_circles import draw_dots_and_lines_on_image
from recalculate_to_latlon import recalculate_coordinates, percentage_to_lat_lon, read_coordinates_from_csv
from config import *

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def find_objects(json_input: str, example_objects: str) -> Tuple[str, List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Identify objects in satellite images using the Molmo VLM model.
    
    Args:
        json_input: JSON string containing object types to search for
        example_objects: Example object format for reference
        
    Returns:
        Tuple containing:
        - JSON string of result coordinates
        - List of percentage coordinates
        - List of lat/lon coordinates
    """
    list_of_the_resulted_coordinates_percentage = []
    list_of_the_resulted_coordinates_lat_lon = []
    
    try:
        processor = AutoProcessor.from_pretrained(
            'cyan2k/molmo-7B-O-bnb-4bit',
            trust_remote_code=True,
            torch_dtype='auto',
            device_map='auto'
        )

        model = AutoModelForCausalLM.from_pretrained(
            'cyan2k/molmo-7B-O-bnb-4bit',
            trust_remote_code=True,
            torch_dtype='auto',
            device_map='auto'
        )
        
        find_objects_json_input = json_input.replace("`", "").replace("json","")
        find_objects_json_input_2 = json.loads(find_objects_json_input)
        
        search_string = ""
        for obj_type in find_objects_json_input_2["object_types"]:
            search_string += obj_type
            
        logger.info(f'Processing {NUMBER_OF_SAMPLES} samples')
        
        for i in range(1, NUMBER_OF_SAMPLES+1):
            logger.info(f'Processing image {i}')
            
            try:
                image_path = f'benchmark-UAV-VLPA-nano-30/images/{i}.jpg'
                inputs = processor.process(
                    images=[Image.open(image_path)],
                    text=f'This is the satellite image of a city. Please, point all the next objects: {search_string}'
                )
                
                inputs = {k: v.to(model.device).unsqueeze(0) for k, v in inputs.items()}
                
                output = model.generate_from_batch(
                    inputs,
                    GenerationConfig(max_new_tokens=2000, stop_strings="<|endoftext|>"),
                    tokenizer=processor.tokenizer
                )
                
                generated_tokens = output[0,inputs['input_ids'].size(1):]
                generated_text = processor.tokenizer.decode(generated_tokens, skip_special_tokens=True)
                
                parsed_points = parse_points(generated_text)
                logger.debug(f'Parsed points for image {i}: {parsed_points}')
                
                csv_file_path = 'benchmark-UAV-VLPA-nano-30/parsed_coordinates.csv'
                coordinates_dict = read_coordinates_from_csv(csv_file_path)
                
                result_coordinates = recalculate_coordinates(parsed_points, i, coordinates_dict)
                
                output_path = f'identified_new_data/identified{i}.jpg'
                draw_dots_and_lines_on_image(image_path, parsed_points, output_path=output_path)
                
                list_of_the_resulted_coordinates_percentage.append(parsed_points)
                list_of_the_resulted_coordinates_lat_lon.append(result_coordinates)
                
            except Exception as e:
                logger.error(f'Error processing image {i}: {str(e)}')
                continue
                
    except Exception as e:
        logger.error(f'Error in find_objects: {str(e)}')
        raise
        
    return json.dumps(result_coordinates), list_of_the_resulted_coordinates_percentage, list_of_the_resulted_coordinates_lat_lon

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
        api_key = os.environ.get("api_key")
        if not api_key:
            raise ValueError("API key not found in environment variables")
            
        llm = ChatOpenAI(
            api_key=api_key,
            model_name='gpt-4',
            temperature=0
        )
        
        # Step 1: Extract object types
        step_1_prompt = PromptTemplate(input_variables=["command"], template=step_1_template)
        step_1_chain = step_1_prompt | llm
        
        # Step 3: Generate flight plan
        step_3_prompt = PromptTemplate(input_variables=["command", "objects"], template=step_3_template)
        step_3_chain = step_3_prompt | llm
        
        object_types_response = step_1_chain.invoke({"command": command})
        object_types_json = object_types_response.content
        
        # Step 2: Find objects on the map
        t1_find_objects = time()
        objects_json, coords_percentage, coords_latlon = find_objects(object_types_json, example_objects)
        t2_find_objects = time()
        del_t_find_objects = (t2_find_objects - t1_find_objects)/60
        
        logger.info(f'Found {len(coords_latlon)} coordinate sets')
        
        # Step 3: Generate flight plans
        t1_generate_drone_mission = time()
        os.makedirs("created_missions", exist_ok=True)
        
        for i, coords in enumerate(coords_latlon, 1):
            flight_plan_response = step_3_chain.invoke({
                "command": command,
                "objects": coords
            })
            
            mission_file = f"created_missions/mission{i}.txt"
            with open(mission_file, "w") as file:
                file.write(str(flight_plan_response.content))
                
            logger.info(f'Generated mission plan {i}')
            
        t2_generate_drone_mission = time()
        del_t_generate_drone_mission = (t2_generate_drone_mission - t1_generate_drone_mission)/60
        
        return flight_plan_response.content, del_t_find_objects, del_t_generate_drone_mission
        
    except Exception as e:
        logger.error(f'Error in generate_drone_mission: {str(e)}')
        raise

def run():
    """Main entry point for the UAV mission generation system."""
    try:
        logger.info('Starting UAV mission generation')
        logger.info(f'CUDA available: {torch.cuda.is_available()}')
        logger.info(f'Processing {NUMBER_OF_SAMPLES} samples')
        
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
    run()