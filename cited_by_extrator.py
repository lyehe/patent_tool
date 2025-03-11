import yaml
from pathlib import Path
import argparse
import re


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


def merge_cited_by_results(results: dict):
    cited_by_results = output_query_results(results, "list_of_forward_citations")
    merged_cited_by_results = []
    for file_name, cited_by_list in cited_by_results.items():
        if cited_by_list:
            merged_cited_by_results.extend(cited_by_list)
    return list(set(merged_cited_by_results))


def extract_patent_numbers(items: list[str]) -> list[str]:
    patent_pattern = r"^[A-Z]{2}\d+[A-Z]\d+$"
    return [item.strip() for item in items if re.match(patent_pattern, item.strip())]


def add_google_patent_urls(patent_numbers: list[str]) -> list[str]:
    base_url = "https://patents.google.com/patent/"
    return [f"{base_url}{patent_number}/en" for patent_number in patent_numbers]


def output_url_to_file(urls: list[str], output_file: Path):
    with open(output_file, "w") as f:
        for url in urls:
            f.write(url + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=Path, required=True)
    parser.add_argument("--output_file", type=Path, required=False)
    args = parser.parse_args()

    if args.output_file:
        output_file = args.output_file
    else:
        output_file = args.input_dir / "cited_by_urls.txt"

    results = extract_from_folder(args.input_dir)
    cited_by = merge_cited_by_results(results)
    patent_numbers = extract_patent_numbers(cited_by)
    urls = add_google_patent_urls(patent_numbers)
    output_url_to_file(urls, output_file)


if __name__ == "__main__":
    main()
