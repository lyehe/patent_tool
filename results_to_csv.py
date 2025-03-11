import yaml
from pathlib import Path
import argparse
import re
import pandas as pd
from typing import Dict, List, Any, Optional, Union


def extract_analysis_from_yaml(yaml_path: Path) -> dict:
    with open(yaml_path, "r") as f:
        data = yaml.safe_load(f)
    return data


def extract_from_folder(folder_path: Path) -> dict:
    results = {}
    for file in folder_path.glob("*.yaml"):
        results[file.stem] = extract_analysis_from_yaml(file)
    return results


def output_query_results(results: dict, key_to_query: str):
    query_results = {}
    for file_name, result in results.items():
        if key_to_query in result:
            query_results[result["patent_number"]] = result[key_to_query]
    return query_results


def extract_patent_numbers(items: List[str]) -> List[str]:
    patent_pattern = r"^[A-Z]{2}\d+[A-Z]\d+$"
    return [item.strip() for item in items if re.match(patent_pattern, item.strip())]


def add_google_patent_urls(patent_numbers: List[str]) -> List[str]:
    base_url = "https://patents.google.com/patent/"
    return [f"{base_url}{patent_number}/en" for patent_number in patent_numbers]


def output_url_to_file(urls: List[str], output_file: Path):
    with open(output_file, "w") as f:
        for url in urls:
            f.write(url + "\n")


def extract_to_csv(
    results: Dict[str, Any], keys_to_extract: List[str], output_file: Path
) -> None:
    """
    Extract specified keys from patent results and output to CSV.
    For list values, output as separate columns (key (1), key (2), etc.) up to 5 items max.
    Always includes patent_number, google_patent_url, title, and assignee as the first columns.
    Preserves the order of input keys.

    :param results: Dictionary of patent results, keyed by filename
    :param keys_to_extract: List of keys to extract from each patent
    :param output_file: Path to save the CSV output
    """
    # Standard columns that always appear first
    standard_columns = ["patent_number", "google_patent_url", "title", "assignee"]
    
    # Preserve the order of input keys, excluding standard columns
    filtered_keys = []
    for k in keys_to_extract:
        if k not in standard_columns and k not in filtered_keys:
            filtered_keys.append(k)

    # Initialize a dictionary to track max items in each list field (capped at 5)
    max_list_lengths = {}
    
    # Track which keys are lists
    list_keys = set()

    # First pass: determine the maximum number of items for each list field (up to 5)
    for _, result in results.items():
        for key in filtered_keys:
            if key in result and isinstance(result[key], list):
                list_keys.add(key)
                current_length = min(len(result[key]), 5)  # Cap at 5 items
                if key not in max_list_lengths or current_length > max_list_lengths[key]:
                    max_list_lengths[key] = current_length

    # Create a list to hold the data
    data = []

    # Process each result
    for file_name, result in results.items():
        # Get patent number and generate Google patent URL
        patent_number = result.get("patent_number", "Unknown")
        google_url = f"https://patents.google.com/patent/{patent_number}/en"

        # Create a dict for this patent with standard columns
        patent_data = {
            "patent_number": patent_number,
            "google_patent_url": google_url,
            "title": result.get("title", "Unknown"),
            "assignee": result.get("assignee", "Unknown"),
        }

        # Extract each requested key
        for key in filtered_keys:
            if key in result:
                value = result[key]
                # Handle special cases like nested dictionaries or lists
                if isinstance(value, dict):
                    # Flatten the dictionary into separate columns
                    for subkey, subvalue in value.items():
                        patent_data[f"{key}_{subkey}"] = subvalue
                elif isinstance(value, list):
                    # Only use up to 5 items from the list
                    limited_value = value[:5]
                    
                    # Create a separate column for each list item (up to 5)
                    for i, item in enumerate(limited_value, 1):
                        patent_data[f"{key} ({i})"] = item
                    
                    # Fill in empty columns for missing items to ensure consistent columns
                    if key in max_list_lengths:
                        for i in range(len(limited_value) + 1, max_list_lengths[key] + 1):
                            patent_data[f"{key} ({i})"] = None
                else:
                    patent_data[key] = value
            else:
                # For non-list keys, add None
                if key not in list_keys:
                    patent_data[key] = None
                else:
                    # For list keys, add numbered None values
                    for i in range(1, max_list_lengths.get(key, 0) + 1):
                        patent_data[f"{key} ({i})"] = None

        data.append(patent_data)

    # Convert to DataFrame and save to CSV
    df = pd.DataFrame(data)
    
    # Build the list of columns in the correct order
    ordered_columns = standard_columns.copy()
    
    # Add remaining columns in the order of filtered_keys
    for key in filtered_keys:
        if key in list_keys:
            # For list keys, add the numbered columns only
            for i in range(1, max_list_lengths.get(key, 0) + 1):
                ordered_columns.append(f"{key} ({i})")
        else:
            # For non-list keys, add the key directly
            ordered_columns.append(key)

    # Only include columns that exist in the DataFrame
    final_columns = [col for col in ordered_columns if col in df.columns]
    
    # Reorder the DataFrame columns
    df = df[final_columns]
    
    df.to_csv(output_file, index=False)
    print(f"Data extracted and saved to {output_file}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=Path, required=True)
    parser.add_argument("--output_file", type=Path, required=False)
    parser.add_argument(
        "--keys", nargs="+", help="Keys to extract from patent data", required=False
    )
    args = parser.parse_args()

    if args.output_file:
        output_file = args.output_file
    else:
        output_file = args.input_dir / "patent_data.csv"

    results = extract_from_folder(args.input_dir)

    if args.keys:
        # Always include standard columns while preserving order of user-specified keys
        standard_keys = ["patent_number", "title", "assignee"]
        
        # First add standard keys if they're not already in the list
        all_keys = []
        for key in standard_keys:
            if key not in all_keys:
                all_keys.append(key)
                
        # Then add user-specified keys while preserving order
        for key in args.keys:
            if key not in all_keys:
                all_keys.append(key)
                
        extract_to_csv(results, all_keys, output_file)
    else:
        # Default behavior - output patent URLs
        cited_patents = []
        for _, result in results.items():
            if "list_of_forward_citations" in result:
                cited_patents.extend(result["list_of_forward_citations"])

        patent_numbers = extract_patent_numbers(cited_patents)
        urls = add_google_patent_urls(patent_numbers)
        url_output_file = args.input_dir / "cited_by_urls.txt"
        output_url_to_file(urls, url_output_file)


if __name__ == "__main__":
    main()
