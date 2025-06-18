#!/usr/bin/env python3

import re
import numpy as np

def parse_ap_results(log_file):
    # Read the log file
    with open(log_file, 'r') as f:
        content = f.read()
    
    # Extract the mAP values (overall results)
    map_pattern = r"mAP: \[([\d\.\s]+)\]"
    map_match = re.search(map_pattern, content)
    overall_maps = np.array([float(x) for x in map_match.group(1).split()])
    
    # Extract per-action AP values
    action_pattern = r"Action: (\d+), AP: \[([\d\.\s]+)\], Mean AP: ([\d.]+)"
    action_matches = re.findall(action_pattern, content)
    
    # Organize the data
    results = {}
    action_means = []
    
    for action, aps, mean_ap in action_matches:
        ap_values = [float(x) for x in aps.split()]
        results[int(action)] = ap_values
        action_means.append(float(mean_ap))
    
    # Calculate the average (already in the log)
    average_ap = np.mean(overall_maps)
    
    return results, overall_maps, action_means, average_ap

def format_to_latex(results, overall_maps, action_means, average_ap):
    tious = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
    
    # Map action numbers to names based on the log file
    action_names = {
        1: "Oral Transit",
        2: "Soft Palate Elevation",
        3: "Hyoid Motion",
        4: "UES Opening",
        5: "Swallow Initiation ",
        6: "Pharyngeal Transit",
        7: "Laryngeal Vestibule Closure"
    }
    
    latex_output = ""
    
    # Add header row
    latex_output += "Action & "
    latex_output += " & ".join([f"tIoU={t:.1f}" for t in tious])
    latex_output += " & Average \\\\\n"
    latex_output += "\n"
    
    # Add action rows
    for action_id in sorted(results.keys()):
        action_name = action_names.get(action_id, f"Action-{action_id}")
        ap_values = results[action_id]
        mean_ap = action_means[action_id-1]  # Adjust for 0-based indexing
        
        latex_output += f"{action_name} & "
        latex_output += " & ".join([f"{ap*100:.1f}" for ap in ap_values])
        latex_output += f" & {mean_ap*100:.1f} \\\\\n"
    
    # Add overall average row with \cell
    latex_output += "\n"
    latex_output += "\cell Average & "
    latex_output += " & ".join([f"\\cell {map_val:.1f}" for map_val in overall_maps])
    latex_output += f" & \\cell {average_ap:.1f} \\\\\n"
    
    return latex_output

def main():
    log_file = "outputs/backbone/2tower_crossmamba_3layer_ep30_vw0.7_heatmap_channelagg_actionformer/eval_0.6.log"
    
    results, overall_maps, action_means, average_ap = parse_ap_results(log_file)
    latex_table = format_to_latex(results, overall_maps, action_means, average_ap)
    
    output_file = "results_table_latex.txt"
    with open(output_file, 'w') as f:
        f.write(latex_table)
    
    print(f"LaTeX table saved to {output_file}")
    print("Preview of the table:")
    print("--------------------")
    print(latex_table)

if __name__ == "__main__":
    main()