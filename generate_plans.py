from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
import json
from transformers import AutoModelForCausalLM, AutoProcessor, GenerationConfig
from PIL import Image
import torch
import os
from parser_for_coordinates import parse_points
from draw_circles import draw_dots_and_lines_on_image
from recalculate_to_latlon import recalculate_coordinates, percentage_to_lat_lon, read_coordinates_from_csv
from time import time
import os
from config import *

def run():
    print(torch.cuda.is_available())
    print('NUMBER_OF_SAMPLES',NUMBER_OF_SAMPLES )
    flight_plan, vlm_model_time, mission_generation_time = generate_drone_mission(command)
    total_computational_time = vlm_model_time + mission_generation_time

    # Evaluation time
    print('-------------------------------------------------------------------')
    print('Time to get VLM results: ', vlm_model_time, 'mins')
    print('Time to get Mission Text files: ', mission_generation_time, 'mins')
    print('Total Computational Time: ', total_computational_time, 'mins')

# 2. Step 2: Use Molmo model to find objects on the map
def find_objects(json_input, example_objects):
    list_of_the_resulted_coordinates_percentage = []
    list_of_the_resulted_coordinates_lat_lon = []
    
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
   
    search_string = str()
    find_objects_json_input = json_input.replace("`", "").replace("json","")    #[9::-3]
    
    find_objects_json_input_2 = json.loads(find_objects_json_input)

    for i in range(0,len(find_objects_json_input_2["object_types"])):
        sample = find_objects_json_input_2["object_types"][i]
        search_string = search_string + sample ##+ ", "

    print('NUMBER_OF_SAMPLES',NUMBER_OF_SAMPLES )
    
    print('\n')
    print('The sample is', sample)
    print('\n')

    for i in range(1, NUMBER_OF_SAMPLES+1):
        print(i)
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

    return json.dumps(result_coordinates), list_of_the_resulted_coordinates_percentage, list_of_the_resulted_coordinates_lat_lon

# Full pipeline function
def generate_drone_mission(command):
    # Insert your key here to run the program
    api_key = os.environ.get("api_key")
    llm = ChatOpenAI(api_key=api_key, 
                 model_name='gpt-4o', temperature=0)

    # 1. Step 1: Extract object types from the user's input command using the LLM
    step_1_prompt = PromptTemplate(input_variables=["command"], template=step_1_template)
    # Instead of using RunnableSequence, we simply use pipe (|)
    step_1_chain = step_1_prompt | llm
    # 3. Step 3: Generate flight plan using LLM and identified objects
    step_3_prompt = PromptTemplate(input_variables=["command", "objects"], template=step_3_template)
    step_3_chain = step_3_prompt | llm
    # Step 1: Extract object types
    
    object_types_response = step_1_chain.invoke({"command": command})

    # Extract the text from the AIMessage object
    object_types_json = object_types_response.content  # Use 'content' to get the actual response text

    # Step 2: Find objects on the map (dummy example for now)
    t1_find_objects = time()
    objects_json, list_of_the_resulted_coordinates_percentage, list_of_the_resulted_coordinates_lat_lon = find_objects(object_types_json, example_objects)
    t2_find_objects = time()

    del_t_find_objects = (t2_find_objects - t1_find_objects)/60

    print('length: ', len(list_of_the_resulted_coordinates_lat_lon))
    
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

if __name__=="__main__":
    run()