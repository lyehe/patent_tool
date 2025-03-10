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
You are a precise JSON formatter. Extract the following information from the Google Patents webpage text while using YAML-style formatting for multi-line text fields.

Follow these strict JSON formatting rules:
1. Use ONLY double quotes for both keys and string values
2. Do not include trailing commas in arrays or objects
3. Ensure all JSON keys are properly quoted
4. Provide your response ONLY as a valid JSON object with no other text before or after
5. If information isn't available, use null or "Not explicitly stated" instead of leaving fields empty
6. For multi-line text fields, format strings in YAML style (clean, indented format)
7. DO NOT include colons (:) within key names - each key must be a simple string
8. For multi-line strings, try to be concise and to the point. Use fewer points if possible. Do not use more than 5 points.
9. DO NOT format the "potential_limitations" field like a key-value pair. It should be formatted as: "potential_limitations": "text describing limitations..." and nothing else.
10. Independent claims should not be dependent on earlier claims. There are usually 1-3 independent claims.
11. The patent_number should be the patent number including the country code.
12. The title should be the complete title of the patent.
13. The number of elements in the list in the example json are just examples. There could be more or less.

{
  "patent_number": "Patent number including country code",
  "title": "Complete title of the patent",
  "inventors": ["Name of inventor 1", "Name of inventor 2"],
  "assignee": "Company or individual that owns the patent rights",
  "type_of_assignee": "Academia or industry",
  "assignee_location": "Country/state of the assignee",
  "dates": {
    "filing_date": "YYYY-MM-DD",
    "publication_date": "YYYY-MM-DD",
    "grant_date": "YYYY-MM-DD or null if not granted",
    "priority_date": "YYYY-MM-DD",
    "expiration_date": "YYYY-MM-DD or null if not available"
  },
  "legal_status": "Current status - active, pending, withdrawn, expired, etc.",
  "category_of_technology": [
    "Main technological field 1",
    "Main technological field 2",
    "Main technological field 3"
  ],
  "one_liner_summary": "One-liner summary of the patent in a sentence or two, including the value proposition",
  "five_keypoints_summary": [
    "Key point 1",
    "Key point 2",
    "Key point 3",
    "Key point 4",
    "Key point 5"
  ],
  "abstract": [
    "Abstract paragraph 1",
    "Abstract paragraph 2",
    "Abstract paragraph 3"
  ],
  "background_summary": [
    "Background point 1",
    "Background point 2",
    "Background point 3"
  ],
  "independent_claims": [
    {
      "claim_number": 1,
      "summary": "Summary of the first independent claim"
    },
    {
      "claim_number": 2,
      "summary": "Summary of the second independent claim"
    }
  ],
  "problem_trying_to_solve": [
    "Problem 1",
    "Problem 2",
    "Problem 3"
  ],
  "key_innovation": [
    "Innovation 1",
    "Innovation 2",
    "Innovation 3"
  ],
  "novelty": [
    "Novelty aspect 1",
    "Novelty aspect 2",
    "Novelty aspect 3"
  ],
  "non_obviousness": [
    "Non-obviousness point 1",
    "Non-obviousness point 2",
    "Non-obviousness point 3"
  ],
  "utility": [
    "Utility aspect 1",
    "Utility aspect 2",
    "Utility aspect 3"
  ],
  "commercial_applications": [
    "Commercial application 1",
    "Commercial application 2",
    "Commercial application 3"
  ],
  "target_application": [
    "Target application 1",
    "Target application 2",
    "Target application 3"
  ],
  "target_users": [
    "User type 1",
    "User type 2"
  ],
  "value_proposition": [
    "Value point 1",
    "Value point 2",
    "Value point 3",
    "Value point 4",
    "Value point 5"
  ],
  "payers": [
    "Payer 1",
    "Payer 2"
  ],
  "market_impact": [
    "Market impact 1",
    "Market impact 2",
    "Market impact 3"
  ],
  "potential_limitations": "Describe limitations or state 'Not explicitly stated' if none found",
  "forward_citations_count": "Number as string",
  "backward_citations_count": "Number as string",
  "list_of_forward_citations": [
    "Patent number 1",
    "Patent number 2",
    "Patent number 3"
  ],
  "details_of_forward_citations": [
    {
      "patent_number": "Patent number 1",
      "title": "Title of citing patent 1",
      "assignee": "Assignee of citing patent 1",
      "year": "Year of citing patent 1"
    },
    {
      "patent_number": "Patent number 2",
      "title": "Title of citing patent 2",
      "assignee": "Assignee of citing patent 2",
      "year": "Year of citing patent 2"
    }
  ],
  "list_of_backward_citations": [
    "Patent number 1",
    "Patent number 2",
    "Non-patent reference"
  ],
  "details_of_backward_citations": [
    {
      "patent_number": "Patent number 1",
      "title": "Title of cited patent 1",
      "assignee": "Assignee of cited patent 1",
      "year": "Year of cited patent 1"
    },
    {
      "patent_number": "Patent number 2",
      "title": "Title of cited patent 2",
      "assignee": "Assignee of cited patent 2",
      "year": "Year of cited patent 2"
    }
  ],
  "related_technologies": [
    "Related technology 1",
    "Related technology 2",
    "Related technology 3"
  ],
  "litigation_history": [
    "Litigation item 1",
    "Litigation item 2"
  ],
  "cpc_classifications": [
    "CPC code 1: Description 1",
    "CPC code 2: Description 2",
    "CPC code 3: Description 3"
  ],
  "drawings_description": [
    "Figure 1: Description 1",
    "Figure 2: Description 2",
    "Figure 3: Description 3"
  ]
}

IMPORTANT: 
1. For ALL array fields (independent_claims, list_of_backward_citations, patents_citing_this, etc.), include ALL relevant items found in the patent document, not just the examples shown in the template.
2. Double check that details_of_backward_citations and details_of_forward_citations are complete.
3. The example arrays show the expected format but not the expected quantity - extract complete data where available.
4. For analytical fields (like market impact), provide an assessment based on the patent content and citation patterns.
5. Return ONLY the JSON object with values filled in from the patent document.
6. Format dates consistently as YYYY-MM-DD
7. Ensure all string values are properly escaped with double quotes
8. For multi-line strings, try to be concise and to the point. Use fewer points if possible. Do not use more than 5 points.
9. DO NOT format the "potential_limitations" field like a key-value pair. It should be formatted as: "potential_limitations": "text describing limitations..." and nothing else.
10. Independent claims should not be dependent on earlier claims. There are usually 1-3 independent claims.
11. The patent_number should be the patent number including the country code.
12. The title should be the complete title of the patent.
13. The number of elements in the list in the example json are just examples. There could be more or less.

Common Errors to Avoid:
- Special characters: Properly escape quotes, less than/greater than symbols, etc.
- Trailing commas: Do not leave trailing commas at the end of arrays or objects
- Inconsistent date formats: Always use YYYY-MM-DD format
- Proper quoting: All keys and string values must be in double quotes
- Unicode characters: Properly escape or convert non-ASCII characters
- Empty values: Use null or "Not explicitly stated" rather than empty strings or fields
- Nested key values: Never format as "key": "value": "description" - this is invalid JSON
- Colons in keys: JSON keys cannot contain colons; use only simple string keys

Example Output Format for Multi-line Text:
"abstract": [
  "This invention relates to a novel compound for treatment of cancer.",
  "The compound selectively targets cancer cells while leaving healthy cells intact.",
  "Clinical trials show efficacy in breast and colon cancers."
]

Example of correct potential_limitations formatting:
"potential_limitations": "The technology requires specialized equipment and has limited depth penetration, so it is not suitable for all cancer types"
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
                print(f"Retrying immediately... (attempt {retries + 1}/{max_retries})")
                # No sleep/wait time here - retry immediately
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
