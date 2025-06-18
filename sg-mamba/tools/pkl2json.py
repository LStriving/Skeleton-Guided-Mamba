import json
import pickle as pkl
import argparse

def pkl2json(pkl_path, json_path, save_pretty=False):
    with open(pkl_path, 'rb') as f:
        data = pkl.load(f)
    # convert result {video-id:[], t-start:[], t-end:[], score:[], label:[]} to
    # {video-id: [{"score": score, "segment":[t-start, t-end], "label": label}, ...]}
    label_dict = {
        1: "OralDelivery",
        2: "SoftPalateLift",
        3: "HyoidExercise",
        4: "UESOpen",
        5: "ThroatSwallow",
        6: "ThroatTransport",
        0: "LaryngealVestibuleClosure"
    }
    data['label'][data['label'] == 7] = 0
    new_results = {}
    for i in range(len(data['video-id'])):
        vid = data['video-id'][i]
        t_start = data['t-start'][i]
        t_end = data['t-end'][i]
        score = data['score'][i]
        label = data['label'][i]
        if vid not in new_results:
            new_results[vid] = []
        new_results[vid].append({
            "score": float(score),
            "segment": [float(t_start), float(t_end)],
            "label": label_dict[label]
        })

    new_results = {
        "results": new_results,
        "version": "1.0",
        "external_data": {}
    }

    # dump results
    with open(json_path, 'w') as f:
        json.dump(new_results, f)
    if save_pretty:
        with open(json_path.replace('.json', '_pretty.json'), 'w') as f:
            json.dump(new_results, f, indent=4)
    
    return new_results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Convert pkl to json')
    parser.add_argument('--pkl_path', type=str, help='path to pkl file')
    parser.add_argument('--json_path', type=str, help='path to save json file')
    parser.add_argument('--save_pretty', action='store_true', help='save pretty json file')
    args = parser.parse_args()
    pkl2json(args.pkl_path, args.json_path, args.save_pretty)