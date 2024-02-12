import csv
import json
import sys
import ast

default_ = {
    'arm': [[0, 15], [15, 25], [25, 70], [70, 90], [90, sys.maxsize]],   
    'leg': [[0, 10], [10, 20], [20, 35], [35, 60], [60, sys.maxsize]],
    'trunk':[[0, 10], [10, 15], [15, 30], [30, 45], [45, sys.maxsize]]     
}

def convert_string_to_dict(s):

    s = s.replace("sys.maxsize", str(sys.maxsize))

    # Initialize an empty dictionary to hold the final result
    result_dict = {}

    # Split the string by lines and iterate through each line
    for line in s.split('\n'):
        if line.strip():  # Check if line is not empty
            # Split the line at '=' to separate the key and the list of lists
            key, value = line.split('=')
            key = key.strip()  # Remove any leading/trailing whitespace from the key
            value = value.strip()  # Remove any leading/trailing whitespace from the value
            # Use ast.literal_eval to safely evaluate the string representation of the list of lists
            result_dict[key] = ast.literal_eval(value)

    return result_dict

def create_json_from_csv(csv_file_path, json_file_path):

    # Initialize an empty dictionary to store the CSV data
    data = {}

    # Open the CSV file for reading
    with open(csv_file_path, mode='r', encoding='utf-8') as csvfile:
        # Create a CSV reader object
        csv_reader = csv.reader(csvfile)
        # Skip the header row
        next(csv_reader, None)
        # Loop through the rows in the CSV
        
        for row in csv_reader:
            
            if len(row) < 4: continue
                
            dict_ = convert_string_to_dict(row[3])
            data[row[0]] = dict_

    data['default'] = default_

    # Open the JSON file for writing
    with open(json_file_path, 'w', encoding='utf-8') as jsonfile:
        # Write the dictionary to the file, formatted as JSON
        json.dump(data, jsonfile, indent=4)

def load_json(json_file_path):

    with open(json_file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)


    for key, value in data.items():
        print(f"Key: {key}, Value: {value}")

if __name__ == "__main__":

    root_dir = "/home/tumeke-balaji/Documents/results/delta/pose-alignment/"

    # Specify the path to your CSV file
    csv_file_path = root_dir + 'delta.csv'

    # Specify the path for the JSON file you want to create
    json_file_path = root_dir + 'cutoff.json'

    create_json_from_csv(csv_file_path, json_file_path)

    load_json(json_file_path)