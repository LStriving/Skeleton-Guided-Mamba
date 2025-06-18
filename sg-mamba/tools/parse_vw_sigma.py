#!/usr/bin/env python3

import re
import numpy as np
import pandas as pd

def parse_sigma_vw_results(log_content):
    # Extract results blocks for each sigma/vw combination
    pattern = r"Processing sigma (\d+) with vw (0\.\d+)\.\.\.(?:.*?\n)+?Avearge mAP: ([\d.]+) \(%\)"
    matches = re.findall(pattern, log_content, re.MULTILINE)
    
    # Organize data for sigma/vw table
    results = []
    for sigma, vw, avg_map in matches:
        results.append({
            "sigma": int(sigma),
            "vw": float(vw),
            "avg_map": float(avg_map)
        })
    
    return results

def find_best_for_each_sigma(results):
    # Group by sigma and find best vw for each
    sigma_best = {}
    for result in results:
        sigma = result["sigma"]
        if sigma not in sigma_best or result["avg_map"] > sigma_best[sigma]["avg_map"]:
            sigma_best[sigma] = result
    
    return sigma_best

def extract_tiou_results(log_content, sigma, vw):
    # Find the specific result block for a given sigma/vw combination
    pattern = r"Processing sigma {} with vw {}\.\.\.(?:[\s\S]*?)mAP: \[([\d\.\s]+)\]".format(sigma, vw)
    match = re.search(pattern, log_content, re.MULTILINE)
    
    if match:
        # Extract mAP values for different tIoU thresholds
        map_values = [float(x) for x in match.group(1).split()]
        return map_values
    
    return None

def create_tables(log_file):
    with open(log_file, 'r') as f:
        log_content = f.read()
    
    # Parse sigma/vw combinations and their average mAP
    results = parse_sigma_vw_results(log_content)
    
    # Create sigma/vw average mAP table
    df_sigma_vw = pd.DataFrame(results)
    pivot_table = df_sigma_vw.pivot(index="sigma", columns="vw", values="avg_map")
    
    # Find best vw for each sigma
    best_configs = find_best_for_each_sigma(results)
    
    # Create table for tIoU thresholds for best configurations
    tiou_thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
    tiou_results = {}
    
    for sigma, config in best_configs.items():
        vw = config["vw"]
        map_values = extract_tiou_results(log_content, sigma, vw)
        if map_values:
            tiou_results[sigma] = map_values
    
    df_tiou = pd.DataFrame(tiou_results, index=tiou_thresholds)
    # Transpose to have tIoU as columns
    df_tiou = df_tiou.T
    df_tiou.index.name = "sigma"
    df_tiou.columns.name = "tIoU"
    # sort the DataFrame by sigma
    df_tiou = df_tiou.sort_index()

    # add average column for tIoU table
    df_tiou["Average"] = df_tiou.mean(axis=1).round(1)

    return pivot_table, df_tiou

def format_latex_table(df, title, with_cell=False):
    """Format DataFrame as LaTeX table with optional \cell for last row"""
    latex = f"% {title}\n"
    
    # Start table
    latex += "\\begin{table}[t]\n"
    latex += "\\centering\n"
    latex += "\\caption{" + title + "}\n"
    
    # Table header
    cols = len(df.columns) + 1
    latex += "\\begin{tabular}{" + "|".join(["c"] * cols) + "}\n"
    latex += "\\hline\n"
    
    # Column headers
    headers = [df.index.name or ""] + [str(col) for col in df.columns]  # Convert column headers to strings
    latex += " & ".join(headers) + " \\\\ \\hline\n"
    
    # Table content
    for i, (idx, row) in enumerate(df.iterrows()):
        # Convert all values to formatted strings
        row_values = [f"{val:.2f}" for val in row]
        row_str = f"{idx} & " + " & ".join(row_values) + " \\\\"
        
        # Add \hline after each row
        if i < len(df) - 1 or not with_cell:
            row_str += " \\hline"
        
        latex += row_str + "\n"
    
    # Add last row with \cell if requested
    if with_cell:
        # Calculate averages for each column
        avgs = df.mean()
        avg_values = [f"\\cell {val:.2f}" for val in avgs]
        avg_row = "Average & " + " & ".join(avg_values) + " \\\\ \\hline"
        latex += avg_row + "\n"
    
    # End table
    latex += "\\end{tabular}\n"
    latex += "\\end{table}\n"
    
    return latex

def main():
    log_file = "outputs/sigma/2tower_crossmamba_sigma.log"
    
    # Create DataFrames for the two tables
    sigma_vw_table, tiou_table = create_tables(log_file)
    
    # Add index and column names
    sigma_vw_table.index.name = "sigma"
    
    # Format as LaTeX tables
    sigma_vw_latex = format_latex_table(sigma_vw_table, "Average mAP for different sigma and vw combinations")
    tiou_latex = format_latex_table(tiou_table, "AP at different tIoU thresholds for best configurations", with_cell=True)
    
    # Save to files
    with open("sigma_vw_table.tex", "w") as f:
        f.write(sigma_vw_latex)
    
    with open("tiou_table.tex", "w") as f:
        f.write(tiou_latex)
    
    # Also print as plain text for easier viewing
    print("Sigma/vw table (average mAP):")
    print(sigma_vw_table.round(2).to_string())
    print("\nBest tIoU table (AP at different thresholds):")
    print(tiou_table.round(2).to_string())

if __name__ == "__main__":
    main()