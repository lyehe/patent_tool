import os
import json
import re
import time
from pathlib import Path
from typing import Any
import yaml  # Requires PyYAML package (pip install pyyaml)
from google import genai
from google.genai import types

SYSTEM_PROMPT = """
You are a precise JSON formatter. Extract the following information from the Google Patents webpage text. 
Create 3 detailed versions of outputs during thinking and consolidate the results to produce the output.
DO NOT SKIP THE 3 VERSIONS, you must first create 3 VERSIONS of the output!

Follow these strict JSON formatting rules:
1. Use ONLY double quotes for both keys and string values
2. Do not include trailing commas in arrays or objects
3. Ensure all JSON keys are properly quoted
4. Provide your response ONLY as a valid JSON object with no other text before or after
5. If information isn't available, use null or "Not explicitly stated" instead of leaving fields empty

{
  "patent_number": "Patent number including country code",
  "title": "Complete title of the patent",
  "inventors": ["Name of inventor 1", "Name of inventor 2"],
  "assignee": "Company or individual that owns the patent rights",
  "type_of_assignee": "Academia or industry",
  "assignee_location": "Country/state of the assignee",
  "dates": {
    "filing_date": "When the patent application was filed",
    "publication_date": "When the patent was published",
    "grant_date": "When the patent was granted, if applicable",
    "priority_date": "Earliest claimed priority date, if applicable",
    "expiration_date": "Estimated expiration date, if available"
  },
  "legal_status": "Current status - active, expired, abandoned, etc.",
  "category_of_technology": ["Main technological field or classification in <=5 bullet points"],
  "one_liner_summary": "One-liner summary of the patent a sentence or two, including the value proposition",
  "five_keypoints_summary": ["Concise summary of the first key point in 5 bullet points"],
  "abstract": ["Concise summary of the abstract text from the patent in <=3 bullet points"],
  "background_summary": ["Concise summary of the background and introduction section in <=3 bullet points"],
  "independent_claims": [
    {
      "claim_number": 1,
      "summary": "Concise summary of the first independent claim"
    },
    {
      "claim_number": "X",
      "summary": "Concise summary of the second independent claim, if available"
    },
    {
      "claim_number": "Y",
      "summary": "Concise summary of the third independent claim, if available"
    }

  ],
  "problem_trying_to_solve": ["Clear description of the problem or limitation the patent addresses in <=3 bullet points"],
  "key_innovation": ["Core technological advancement or novel solution introduced in <=3 bullet points"],
  "novelty": ["What makes this invention new compared to prior art in <=3 bullet points"],
  "non_obviousness": ["Why this invention would not be obvious to a person skilled in the art in <=3 bullet points"],
  "utility": ["Practical applications and usefulness of the invention in <=3 bullet points"],
  "commercial_applications": ["Potential or existing commercial implementations in bullet points"],
  "target_application": ["Specific disease / type of surgery in bullet points"],
  "target_users": ["Top 1 or 2 target users of the technology i bullet points"],
  "value_proposition": ["Pains/gains addressed by the technology in <=5 bullet points"],
  "payers": ["Top 1 or 2 decision makers for purchasing a product using this technology in bullet points"],
  "market_impact": ["Assessment of potential market impact or importance in <=3 bullet points"],
  "potential_limitations": "Weaknesses or limitations of the patent claims in <=3 bullet points",
  "forward_citations_count": "Number of patents citing this patent",
  "backward_citations_count": "Number of patents cited by this patent",
  "list_of_forward_citations": [
    "patent_number_1", 
    "patent_number_2", 
    "additional_forward_citations",
  ],
  "details_of_forward_citations": [
    {
      "patent_number": "patent_number_1",
      "title": "Title of citing patent 1",
      "assignee": "Assignee of citing patent 1",
      "year": "Year of citing patent 1"
    },
    {
      "patent_number": "patent_number_2",
      "title": "Title of citing patent 2",
      "assignee": "Assignee of citing patent 2",
      "year": "Year of citing patent 2"
    },
  ],
  "list_of_backward_citations": [
    "patent_number_1", 
    "patent_number_2", 
    "non_patent_reference",
    "additional_backward_citations",
  ],
  "details_of_backward_citations": [
    {
      "patent_number": "patent_number_1",
      "title": "Title of cited patent 1",
      "assignee": "Assignee of cited patent 1",
      "year": "Year of cited patent 1"
    },
    {
      "patent_number": "patent_number_2",
      "title": "Title of cited patent 2",
      "assignee": "Assignee of cited patent 2",
      "year": "Year of cited patent 2"
    },
  ],
  "related_technologies": ["Other technologies this patent relates to"],
  "litigation_history": ["Any litigation involving this patent, if available"],
  "cpc_classifications": ["Cooperative Patent Classification codes with brief descriptions"],
  "drawings_description": ["Brief description of the most important figures/drawings"]
}
IMPORTANT: For ALL array fields (independent_claims, list_of_backward_citations, patents_citing_this, list_of_prior_art_citations, etc.), include ALL relevant items found in the patent document, not just the  shown in the template examples. 
Double check to make sure all the details_of_backward_citations and details_of_forward_citations are complete.
The example arrays show the expected format but not the expected quantity - extract complete data where available.
For analytical fields (like market impact), provide an assessment based on the patent content and citation patterns. 
Return ONLY the JSON object with values filled in from the patent document.

Avoid JSON Errors and Parsing Issues:
- Inconsistent String Formatting:
  - Some strings contain special characters like quotes within quotes that aren't properly escaped
  - For example, in citations where titles contain quotes (e.g., "Integrin v6: Structure, function and role in health and disease")
- Date Format Inconsistencies:
  - Some patent has dates in uncommon format (e.g., "20110713" instead of "2011-07-13")
  - Some dates are strings with quotes, others are null, others are without quotes
  - Some patent numbers have commas in them (e.g., "EP000,0000, A1" instead of "EP0000000A1")
- Structure Problems with Arrays and Objects:
  - In some cases, arrays may have trailing commas which are invalid in JSON
  - Nested object structures may have inconsistent formatting
- Incorrect Unicode Characters:
  - Characters like < and > in patent titles (e.g., "alpha<v>beta<3>") aren't properly escaped
  - Non-ASCII characters in names and titles (e.g., "Mnchen" with umlauts)


"""


def fix_json_string(json_str: str) -> str:
    """Attempt to fix common JSON formatting issues.

    :param json_str: The potentially malformed JSON string
    :return: A hopefully corrected JSON string
    """
    # Find the JSON part - sometimes the model outputs text before/after the JSON
    json_match = re.search(r"({[\s\S]*})", json_str)
    if json_match:
        json_str = json_match.group(1)

    # Fix missing quotes around keys
    json_str = re.sub(r"([{,])\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*:", r'\1"\2":', json_str)

    # Fix single quotes to double quotes (but be careful with nested quotes)
    json_str = re.sub(r"(?<=[,{[\s])\'([^\']*?)\'(?=[,}\]:])", r'"\1"', json_str)

    # Fix trailing commas in arrays/objects
    json_str = re.sub(r",\s*([}\]])", r"\1", json_str)

    return json_str


def parse_json_safely(json_str: str) -> dict[str, Any] | None:
    """Parse JSON with multiple fallback methods.

    :param json_str: JSON string to parse
    :return: Parsed dictionary or None if all parsing attempts fail
    """
    # First attempt: direct parsing
    try:
        return json.loads(json_str)
    except json.JSONDecodeError:
        pass

    # Second attempt: try fixing common issues
    try:
        fixed_json = fix_json_string(json_str)
        return json.loads(fixed_json)
    except json.JSONDecodeError:
        pass

    # Third attempt: use a more permissive approach with ast.literal_eval
    try:
        import ast

        # Replace "true", "false", "null" with Python equivalents
        python_str = (
            json_str.replace("true", "True")
            .replace("false", "False")
            .replace("null", "None")
        )
        return ast.literal_eval(python_str)
    except (SyntaxError, ValueError):
        return None


def generate(input_file_path: str | Path) -> dict[str, Any]:
    """Process the entire text from a file using Gemini API and return result as a dictionary.

    :param input_file_path: Path to the input text file
    :return: Dictionary containing the parsed JSON response
    """
    # Read input from file
    with open(input_file_path, "r", encoding="utf-8") as file:
        input_text = file.read()

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY environment variable is not set")

    client = genai.Client(api_key=api_key)

    model = "gemini-2.0-flash-thinking-exp-01-21"
    contents = [
        types.Content(
            role="user",
            parts=[
                types.Part.from_text(text=input_text),
            ],
        ),
    ]
    generate_content_config = types.GenerateContentConfig(
        temperature=0.25,
        top_p=0.75,
        top_k=64,
        max_output_tokens=65536,
        safety_settings=[
            types.SafetySetting(
                category="HARM_CATEGORY_CIVIC_INTEGRITY",
                threshold="OFF",  # Off
            ),
        ],
        response_mime_type="text/plain",
        system_instruction=[
            types.Part.from_text(text=SYSTEM_PROMPT),
        ],
    )

    # Collect the full response
    full_response = ""
    for chunk in client.models.generate_content_stream(
        model=model,
        contents=contents,
        config=generate_content_config,
    ):
        full_response += chunk.text
        print(".", end="", flush=True)  # Progress indicator
    print()  # Newline after progress indicators

    # Try to parse the JSON response with our robust parser
    result_dict = parse_json_safely(full_response)

    if result_dict is None:
        # If all parsing attempts fail, save the raw response for debugging
        error_file = Path(input_file_path).stem + "_error_response.txt"
        with open(error_file, "w", encoding="utf-8") as f:
            f.write(full_response)
        raise ValueError(
            f"Failed to parse response as JSON. Raw response saved to {error_file}"
        )

    return result_dict


def save_as_yaml(data: dict[str, Any], output_file_path: str | Path) -> None:
    """Save dictionary data as YAML file.

    :param data: Dictionary containing the data to save
    :param output_file_path: Path to save the YAML file
    """
    # Ensure directory exists
    output_path = Path(output_file_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as file:
        yaml.dump(data, file, default_flow_style=False, sort_keys=False)


def process_file_with_retry(
    input_file: Path, output_dir: Path, max_retries: int = 3
) -> bool:
    """Process a single file with retry logic.

    :param input_file: Path to the input text file
    :param output_dir: Directory to save the output
    :param max_retries: Maximum number of retry attempts
    :return: True if processing succeeded, False otherwise
    """
    retries = 0
    output_file = output_dir / f"{input_file.stem}_result.yaml"

    # Skip if output already exists (comment out this section if you want to reprocess)
    if output_file.exists():
        print(f"Output file {output_file} already exists, skipping.")
        return True

    while retries < max_retries:
        try:
            print(f"Processing {input_file} (attempt {retries + 1}/{max_retries})...")
            patent_data = generate(input_file)

            # Save result
            save_as_yaml(patent_data, output_file)

            print(f"Successfully processed {input_file}")
            print(f"Result saved to {output_file}")
            return True

        except Exception as e:
            retries += 1
            print(f"Error processing {input_file}: {e}")

            if retries < max_retries:
                wait_time = 10 * retries  # Exponential backoff: 10s, 20s, 30s...
                print(f"Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
            else:
                print(f"Failed to process {input_file} after {max_retries} attempts")
                return False


def batch_process_folder(input_dir: str | Path, max_retries: int = 3) -> list[Path]:
    """Process all text files in a folder.

    :param input_dir: Directory containing input text files
    :param max_retries: Maximum number of retry attempts per file
    :return: List of files that failed to process
    """
    input_path = Path(input_dir)
    print(input_path.absolute())

    # Create output directory
    output_dir = input_path.parent / f"{input_path.name}_results"
    output_dir.mkdir(exist_ok=True)

    # Get all text files in the directory
    text_files = [
        f for f in input_path.iterdir() if f.is_file() and f.suffix.lower() in [".txt"]
    ]

    if not text_files:
        print(f"No text files found in {input_dir}")
        return []

    print(f"Found {len(text_files)} text files to process")

    failed_files = []
    for idx, file in enumerate(text_files, 1):
        print(f"\nProcessing file {idx}/{len(text_files)}: {file.name}")
        success = process_file_with_retry(file, output_dir, max_retries)

        if not success:
            failed_files.append(file)

    # Summary
    total = len(text_files)
    succeeded = total - len(failed_files)
    print(f"\nProcessing complete: {succeeded}/{total} files successfully processed")

    if failed_files:
        print("Failed files:")
        for file in failed_files:
            print(f"  - {file}")

    return failed_files


if __name__ == "__main__":
    import sys
    import argparse

    parser = argparse.ArgumentParser(
        description="Process patent text files with Gemini API"
    )
    parser.add_argument(
        "--folder", "-f", type=str, help="Folder containing input text files"
    )
    parser.add_argument("--file", type=str, help="Single input text file to process")
    parser.add_argument(
        "--retries", type=int, default=3, help="Maximum retry attempts for each file"
    )

    args = parser.parse_args()

    # Verify API key
    if not os.environ.get("GEMINI_API_KEY"):
        print("Error: GEMINI_API_KEY environment variable is not set")
        print("Please set it with:")
        print("  export GEMINI_API_KEY=your_api_key  # On Linux/macOS")
        print("  set GEMINI_API_KEY=your_api_key     # On Windows cmd")
        print("  $env:GEMINI_API_KEY='your_api_key'  # On Windows PowerShell")
        sys.exit(1)

    try:
        if args.folder:
            # Process folder
            batch_process_folder(args.folder, args.retries)
        elif args.file:
            # Process single file
            input_file = Path(args.file)
            output_dir = input_file.parent / f"{input_file.parent.name}_results"
            output_dir.mkdir(exist_ok=True)
            process_file_with_retry(input_file, output_dir, args.retries)
        else:
            # Default folder
            default_folder = Path("patent_data/text/raw")
            if default_folder.exists() and default_folder.is_dir():
                batch_process_folder(default_folder, args.retries)
            else:
                print(
                    "Error: No input folder or file specified and default folder not found"
                )
                parser.print_help()
                sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
