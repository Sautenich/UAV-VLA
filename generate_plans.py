from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
import json
from transformers import AutoModelForCausalLM, AutoProcessor, GenerationConfig
from PIL import Image
import requests
import torch
import os
import re
import csv
from parser_for_coordinates import parse_points
from draw_circles import draw_dots_and_lines_on_image
from recalculate_to_latlon import recalculate_coordinates, percentage_to_lat_lon, read_coordinates_from_csv
from time import time

#os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'


list_of_the_resulted_coordinates_percentage = []
list_of_the_resulted_coordinates_lat_lon = []

#import molmo_inference
# Initialize the ChatGPT-4 model using ChatOpenAI

print(torch.cuda.is_available())

LIST_OF_ANSWERS = []

NUMBER_OF_SAMPLES = 30 #len(os.listdir('/VLM_Drone/dataset_images'))
print('NUMBER_OF_SAMPLES',NUMBER_OF_SAMPLES )

# load the processor
processor = AutoProcessor.from_pretrained(
    'cyan2k/molmo-7B-O-bnb-4bit',
    trust_remote_code=True,
    torch_dtype='auto',
    device_map='auto'
)

# load the model
model = AutoModelForCausalLM.from_pretrained(
    'cyan2k/molmo-7B-O-bnb-4bit',
    trust_remote_code=True,
    torch_dtype='auto',
    device_map='auto'
)


llm = ChatOpenAI(api_key='', model_name='gpt-4o', temperature=0)

# 1. Step 1: Extract object types from the user's input command using the LLM

step_1_template = """
Extract all types of objects the drone needs to find from the following mission description:
"{command}"

Output the result in JSON format with a list of object types.
Example output:
{{
    "object_types": ["village", "airfield", "stadium", "tennis court", "building", "ponds", "crossroad", "roundabout"]
}}
"""

step_1_prompt = PromptTemplate(input_variables=["command"], template=step_1_template)

# Instead of using RunnableSequence, we simply use pipe (|)
step_1_chain = step_1_prompt | llm

##print(step_1_chain)


example_objects = '''
{
        "village_1": {"type": "village", "coordinates": [1.5, 3.5]},
        "village_2": {"type": "village", "coordinates": [2.5, 6.0]},
        "airfield": {"type": "airfield", "coordinates": [8.0, 6.5]}
    }
'''


# 2. Step 2: Use Molmo model to find objects on the map
def find_objects(json_input, example_objects):
    """
    Placeholder for SAM model to process input object types and output valid objects.
    For now, return a dummy dictionary of identified objects.
    """
    search_string = str()
    find_objects_json_input = json_input.replace("`", "").replace("json","")    #[9::-3]
    
    find_objects_json_input_2 = json.loads(find_objects_json_input)

    for i in range(0,len(find_objects_json_input_2["object_types"])):
        #print(find_objects_json_input_2["object_types"][i]) #Show in console the type of an object
        sample = find_objects_json_input_2["object_types"][i]
        search_string = search_string + sample ##+ ", "


  
    print('\n')
    print('The sample is', sample)
    print('\n')

    for i in range(1, NUMBER_OF_SAMPLES+1):
        print(i)
        string = 'benchmark-UAV-VLPA-nano-30/images/' + str(i) + '.jpg' 
    #process the image and text
        inputs = processor.process(
            images=[Image.open('benchmark-UAV-VLPA-nano-30/images/' + str(i) + '.jpg')],
            text=f'''
            This is the satellite image of a city. Please, point all the next objects: {sample} 
            '''
        )

        



    #move inputs to the correct device and make a batch of size 1
        inputs = {k: v.to(model.device).unsqueeze(0) for k, v in inputs.items()}

        #generate output; maximum 200 new tokens; stop generation when <|endoftext|> is generated
        output = model.generate_from_batch(
            inputs,
            GenerationConfig(max_new_tokens=2000, stop_strings="<|endoftext|>"),
            tokenizer=processor.tokenizer
        )

        #only get generated tokens; decode them to text
        generated_tokens = output[0,inputs['input_ids'].size(1):]
        generated_text = processor.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        

        #print the generated text
        ##print('molmo_output =', generated_text)

        parsed_points = parse_points(generated_text)
       
        print('\n')
        print(parsed_points)
        print('\n')

        image_number = i

        csv_file_path = 'benchmark-UAV-VLPA-nano-30/parsed_coordinates.csv'
        coordinates_dict = read_coordinates_from_csv(csv_file_path)

        result_coordinates = recalculate_coordinates(parsed_points, image_number, coordinates_dict)
        draw_dots_and_lines_on_image(f'benchmark-UAV-VLPA-nano-30/images/{i}.jpg', parsed_points, output_path=f'identified_new_data/identified{i}.jpg')

        print(result_coordinates)

        list_of_the_resulted_coordinates_percentage.append(parsed_points)
        list_of_the_resulted_coordinates_lat_lon.append(result_coordinates)



    #object_types = json.loads(json_input)["object_types"]

    return json.dumps(result_coordinates), list_of_the_resulted_coordinates_percentage, list_of_the_resulted_coordinates_lat_lon

# 3. Step 3: Generate flight plan using LLM and identified objects
step_3_template = """
Given the mission description: "{command}" and the following identified objects: {objects}, generate a flight plan in pseudo-language.

The available commands are on the website:


Some hints:
- arm throttle: arm the copter
- takeoff Z: lift Z meters
- disarm: disarm the copter
- mode rtl: return to home
- mode circle: circle and observe at the current position
- mode guided(X Y Z): fly to the specified location

Use the identified objects to create the mission.

Provide me only with commands string-by-string.


Example output:


arm throttle
mode guided 43.237763722222226 -85.79224314444444 100
mode guided 43.237765234234234 -85.79224314235235 100
mode circle
mode rtl
disarm


"""

step_3_prompt = PromptTemplate(input_variables=["command", "objects"], template=step_3_template)
step_3_chain = step_3_prompt | llm

# Full pipeline function
def generate_drone_mission(command):
    # Step 1: Extract object types
    object_types_response = step_1_chain.invoke({"command": command})
    
    #print('object_types_response =', object_types_response)

    
    # Extract the text from the AIMessage object
    object_types_json = object_types_response.content  # Use 'content' to get the actual response text

    # Step 2: Find objects on the map (dummy example for now)
    t1_find_objects = time()
    objects_json, list_of_the_resulted_coordinates_percentage, list_of_the_resulted_coordinates_lat_lon = find_objects(object_types_json, example_objects)
    t2_find_objects = time()

    del_t_find_objects = (t2_find_objects - t1_find_objects)/60

    print('length: ', len(list_of_the_resulted_coordinates_lat_lon))


    #print('objects_json =', objects_json)

    
    # Step 3: Generate the flight plan
    t1_generate_drone_mission = time()

    for i in range(1,len(list_of_the_resulted_coordinates_lat_lon)+1): 
        flight_plan_response = step_3_chain.invoke({"command": command, "objects": list_of_the_resulted_coordinates_lat_lon[i-1]})
    #print('flight_plan_response = ', flight_plan_response)
        with open(f"created_missions/mission{i}.txt","w") as file:   
            file.write(str(flight_plan_response.content))

        print(flight_plan_response.content)

    t2_generate_drone_mission = time()
    del_t_generate_drone_mission = (t2_generate_drone_mission - t1_generate_drone_mission)/60


    return flight_plan_response.content, del_t_find_objects, del_t_generate_drone_mission  # Return the response text from AIMessage

# Example usage
command = """Create a flight plan for the quadcopter to fly around each of the building at the height 100m return to home and land at the take-off point."""


# Run the full pipeline

flight_plan, vlm_model_time, mission_generation_time = generate_drone_mission(command)
total_computational_time = vlm_model_time + mission_generation_time

# Evaluation time
print('-------------------------------------------------------------------')
print('Time to get VLM results: ', vlm_model_time, 'mins')
print('Time to get Mission Text files: ', mission_generation_time, 'mins')
print('Total Computational Time: ', total_computational_time, 'mins')
