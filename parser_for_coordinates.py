import xml.etree.ElementTree as ET

def parse_points(points_str):
    # Fix the input string by ensuring it's properly formatted
    fixed_str = points_str.strip()
    
    # Parse the XML string
    root = ET.fromstring(fixed_str)

    # Initialize an empty dictionary for the output
    output = {}
    
    # Extract the building type from the alt attribute
    building_type = root.attrib.get('alt', '').strip().split(',')[0] if 'alt' in root.attrib else 'building'

    # Extract coordinates from attributes
    for i in range(1, 50):  # Assuming x1 to x5 and y1 to y5
        x_attr = f"x{i}"
        y_attr = f"y{i}"
        
        if x_attr in root.attrib and y_attr in root.attrib:
            x = float(root.attrib[x_attr])
            y = float(root.attrib[y_attr])
            output[f"{building_type}_{i}"] = {"type": building_type, "coordinates": [x, y]}

    return output


if __name__== "__main__":
    # Example input
    points_string = '<points x1="40.2" y1="39.5" x2="47.7" y2="39.0" x3="64.9" y3="41.2" x4="65.2" y4="87.9" x5="80.2" y5="20.7" x6="80.2" y6="20.7" alt="building, ">building, </points>'
    # Parsing the points
    result = parse_points(points_string)
    print(result)