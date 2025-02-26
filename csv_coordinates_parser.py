import csv
import re

def parse_csv(input_data, output_filename):
    # Regex pattern to capture all required data
    pattern = r"Image: (\d+\.jpg)\s+NW Corner Lat: ([\d.-]+),\s+NW Corner Long: ([\d.-]+),\s+SE Corner Lat: ([\d.-]+),\s+SE Corner Long: ([\d.-]+)"

    # Find all matches
    matches = re.findall(pattern, input_data)

    # Prepare the CSV data
    csv_data = [["Image", "NW Corner Lat", "NW Corner Long", "SE Corner Lat", "SE Corner Long"]]

    # Add the extracted information to the CSV data
    for match in matches:
        csv_data.append([match[0], match[1], match[2], match[3], match[4]])

    # Write to CSV file
    with open(output_filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(csv_data)

    print(f"CSV file '{output_filename}' created successfully.")

def run_parser():
    # Example input string
    data_string = open("benchmark-UAV-VLPA-nano-30/img_lat_long_data.txt", "r")
    # print(data_string.read())
    data_string = str(data_string.read()) 
    print(data_string)
    # Output CSV filename
    output_filename = 'benchmark-UAV-VLPA-nano-30/parsed_coordinates.csv'

    # Run the parser
    # parse_to_csv(data_string, output_filename)
    parse_csv(data_string, output_filename)