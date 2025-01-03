import csv
import re

def parse_to_csv(data_string, output_filename):
    # Split the input string into sections for each image
    images = data_string.strip().split('---')

    # Prepare a list to hold the parsed data
    parsed_data = []

    # Define a regex pattern to capture the relevant data
    pattern = re.compile(
        r'Image:\s*(\d+\.jpg)\s*'
        r'NW Corner Lat:\s*([-\d.]+),\s*'
        r'NW Corner Long:\s*([-\d.]+),\s*'
        r'SE Corner Lat:\s*([-\d.]+),\s*'
        r'SE Corner Long:\s*([-\d.]+)'
    )

    for image in images:
        match = pattern.search(image)
        if match:
            image_name = match.group(1)
            nw_lat = match.group(2)
            nw_long = match.group(3)
            se_lat = match.group(4)
            se_long = match.group(5)

            # Append the data to the list
            parsed_data.append({
                'Image': image_name,
                'NW Corner Lat': nw_lat,
                'NW Corner Long': nw_long,
                'SE Corner Lat': se_lat,
                'SE Corner Long': se_long
            })

    # Write the parsed data to a CSV file
    with open(output_filename, mode='w', newline='') as csv_file:
        fieldnames = ['Image', 'NW Corner Lat', 'NW Corner Long', 'SE Corner Lat', 'SE Corner Long']
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)

        writer.writeheader()  # Write the header
        writer.writerows(parsed_data)  # Write the data rows

# Example input string
data_string = """
Image: 1.jpg
NW Corner Lat: 42.884005555555554,
NW Corner Long: -85.76918888888889,
SE Corner Lat: 42.87726111111111,
SE Corner Long: -85.75970833333334

---
Image: 2.jpg
NW Corner Lat: 42.918299999999995,
NW Corner Long: -85.7699566666666,
SE Corner Lat: 42.91155555555555,
SE Corner Long: -85.76048055555556

---
Image: 3.jpg
NW Corner Lat: 43.24042777777778,
NW Corner Long: -85.79607777777778,
SE Corner Lat: 43.23368333333333,
SE Corner Long: -85.78653888888888

---
Image: 4.jpg
NW Corner Lat: 43.29844444444444,
NW Corner Long: -85.51565833333333,
SE Corner Lat: 43.29168055555555,
SE Corner Long: -85.5061416666666

---
Image: 5.jpg
NW Corner Lat: 42.778375,
NW Corner Long: -85.78739722222222,
SE Corner Lat: 42.76762777777776,
SE Corner Long: -85.74793333333334

---
Image: 6.jpg
NW Corner Lat: 42.76861111111111,
NW Corner Long: -85.66414722222223,
SE Corner Lat: 42.761858333333336,
SE Corner Long: -85.65469444444444

---
Image: 7.jpg
NW Corner Lat: 42.90241111111111,
NW Corner Long: -85.3123361111111,
SE Corner Lat: 42.89562777777776,
SE Corner Long: -85.30290277777777

---
Image: 8.jpg
NW Corner Lat: 42.85439444444445,
NW Corner Long: -85 3116027777778,
SE Corner Lat: 42.84761111111114,
SE Corner Long: -85.30217777777777

---
Image: 9.jpg
NW Corner Lat: 42.96056111111111,
NW Corner Long: -85.6775138888889,
SE Corner Lat: 42.953808333333335,
SE Corner Long: -85.66803333833334

---
Image: 10.jpg
NW Corner Lat: 43.02106944444444,
NW Corner Long: -85.78165277777778,
SE Corner Lat: 43.014325,
SE Corner Long: -85.77214722222222

---
Image: 11.jpg
NW Corner Lat: 42.88389166666666,
NW Corner Long: -85.77851666660565,
SE Corner Lat: 42.87714722222222,
SE Corner Long: -85.76903833833333

---
Image: 12.jpg
NW Corner Lat: 42.88489444444444,
NW Corner Long: -85.69457222222222,
SE Corner Lat: 42.878144444444445,
SE Corner Long: -85.6851

---
Image: 13.jpg
NW Corner Lat: 42.89904166666667,
NW Corner Long: -85.65755,
SE Corner Lat: 42.89228611111111,
SE Corner Long: -85.64807777777779

---
Image: 14.jpg
NW Corner Lat: 43.14564166666666,
NW Corner Long: -85.69081944444444,
SE Corner Lat: 43.138888888888806,
SE Corner Long: 85.68130555555555

---
Image: 15.jpg
NW Corner Lat: 42.81619722222222,
NW Corner Long: -85.70241944444444,
SE Corner Lat: 42.80944444444445,
SE Corner Long: -85.6929555555556

---
Image: 16.jpg
NW Corner Lat: 43.09763055555556,
NW Corner Long: -85.6897888888889,
SE Corner Lat: 43.09087777777778,
SE Corner Long: -85.68028333333334

---
Image: 17.jpg
NW Corner Lat: 43.110686111111114,
NW Corner Long: -85.74625277777778,
SE Corner Lat: 43.10393888888889,
SE Corner Long: -85.7367388888889

---
Image: 18.jpg
NW Corner Lat: 43.13812222222222,
NW Corner Long: -85.74686666666666,
SE Corner Lat: 43.131375,
SE Corner Long: -85.73734722222223

---
Image: 19.jpg
NW Corner Lat: 43.076505555555556,
NW Corner Long: -85.73613055555556,
SE Corner Lat: 43.06975893333334,
SE Corner Long: -85.72662222222222

---
Image: 20.jpg
NW Corner Lat: 43.15206111111111,
NW Corner Long: -85.72843888888869,
SE Corner Lat: 43.145313888886806,
SE Corner Long: -85.71891944444445

---
Image: 21.jpg
NW Corner Lat: 43.179608333333364,
NW Corner Long: -85.71967222222223,
SE Corner Lat: 43.17285893333333,
SE Corner Long: -85.71015

---
Image: 22.jpg
NW Corner Lat: 42.857350000000004,
NW Corner Long: -85.70330883833334,
SE Corner Lat: 42.8506,
SE Corner Long: -85.69363611111111

---
Image: 23.jpg
NW Corner Lat: 42.78200833333333,
NW Corner Long: -85.69236044444445,
SE Corner Lat: 42.77525893333333,
SE Corner Long: -85.68291111111111

---
Image: 24.jpg
NW Corner Lat: 43.09066388888889,
NW Corner Long: -85.699,
SE Corner Lat: 43.083913888888804,
SE Corner Long: -85.68949444444445

---
Image: 25.jpg
NW Corner Lat: 42.80966111111111,
NW Corner Long: -85.67432500000001,
SE Corner Lat: 42.80290833333333,
SE Corner Long: -85.66486388888889

---
Image: 26.jpg
NW Corner Lat: 42.8028,
NW Corner Long: -85.67417777777779,
SE Corner Lat: 42.79604722222222,
SE Corner Long: -85.66471944444444

---
Image: 27.jpg
NW Corner Lat: 43.056691666656666,
NW Corner Long: -85.67020000000001,
SE Corner Lat: 43.04993888888889,
SE Corner Long: -85.66070277777779

---
Image: 28.jpg
NW Corner Lat: 43.2697305S5555554,
NW Corner Long: -85.63715388886869,
SE Corner Lat: 43.262975,
SE Corner Long: -85.6276361111111

---
Image: 29.jpg
NW Corner Lat: 42.80342777777778,
NW Corner Long: -85.61828888888888,
SE Corner Lat: 42.79686944444444,
SE Corner Long: -85.6088361111111

---
Image: 30.jpg
NW Corner Lat: 43.19438888888889,
NW Corner Long: -85.62623333333333,
SE Corner Lat: 43.19438888888889,
SE Corner Long: -85.61685833333333

---
Image: 31.jpg
NW Corner Lat: 43.112186111111114,
NW Corner Long: -85.61518888888888,
SE Corner Lat: 43.10542777777778,
SE Corner Long: -85.60569166666666

---
Image: 32.jpg
NW Corner Lat: 43.11904444444445,
NW Corner Long: -85.61532777777778,
SE Corner Lat: 43.11228611111111,
SE Corner Long: -85.60582777777778

---
Image: 33.jpg
NW Corner Lat: 43.16019722222222,
NW Corner Long: -85.61618388888868,
SE Corner Lat: 43.153438888886806,
SE Corner Long: -85.6066S5S5S5S555

---
Image: 34.jpg
NW Corner Lat: 42.83106666666667,
NW Corner Long: -85.60020277777777,
SE Corner Lat: 42.82430555555556,
SE Corner Long: -85.59075

---
Image: 35.jpg
NW Corner Lat: 43.119147222222225,
NW Corner Long: -85.60596666666568,
SE Corner Lat: 43.11238888888889,
SE Corner Long: -85.S9645666660565

---
Image: 36.jpg
NW Corner Lat: 43.215260444444445,
NW Corner Long: -85.59852222222221,
SE Corner Lat: 43.208511111111115,
SE Corner Long: -85.58900883833333

---
Image: 37.jpg
NW Corner Lat: 42.97530833333334,
NW Corner Long: -85.58439722222222,
SE Corner Lat: 42.96854722222223,
SE Corner Long: -85.57492222222221

---
Image: 38.jpg
NW Corner Lat: 43.208511111111115,
NW Corner Long: -85.58900883833333,
SE Corner Lat: 43.201750000000004,
SE Corner Long: -85.57949722222222

---
"""

# Output CSV filename
output_filename = 'parsed_coordinates.csv'

# Run the parser
parse_to_csv(data_string, output_filename)
