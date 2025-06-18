import json
import pickle as pkl
import argparse
import numpy as np

def json2pkl(json_path, pkl_path):
    # Load JSON data
    with open(json_path, 'r') as f:
        json_data = json.load(f)
    
    # Extract results from JSON
    results = json_data.get('results', {})
    
    # Reverse label dictionary to map string labels back to integers
    label_dict = {
        "OralDelivery": 1,
        "SoftPalateLift": 2,
        "HyoidExercise": 3,
        "UESOpen": 4,
        "ThroatSwallow": 5,
        "ThroatTransport": 6,
        "LaryngealVestibuleClosure": 0
    }
    
    # Initialize data structure with lists to collect entries
    data = {
        'video-id': [],
        't-start': [],
        't-end': [],
        'score': [],
        'label': []
    }
    
    # Iterate over each video and its entries in the results
    for video_id, entries in results.items():
        for entry in entries:
            data['video-id'].append(video_id)
            segment = entry['segment']
            data['t-start'].append(segment[0])
            data['t-end'].append(segment[1])
            data['score'].append(entry['score'])
            label_str = entry['label']
            data['label'].append(label_dict[label_str])
    
    # Convert lists to numpy arrays to match original PKL structure
    for key in data:
        data[key] = np.array(data[key])
    
    # Save the reconstructed data to PKL file
    with open(pkl_path, 'wb') as f:
        pkl.dump(data, f)
    
    return data

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Convert pkl to json')
    parser.add_argument('--pkl_path', type=str, help='path to pkl file')
    parser.add_argument('--json_path', type=str, help='path to save json file')
    args = parser.parse_args()
    json2pkl(args.json_path, args.pkl_path)