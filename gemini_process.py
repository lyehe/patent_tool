import os
import json
import re
from pathlib import Path
from typing import Any
import yaml  # Requires PyYAML package (pip install pyyaml)
from google import genai
from google.genai import types
from json_repair import repair_json
from datetime import datetime

SYSTEM_PROMPT = """
You are a precise JSON formatter tasked with extracting structured patent information. Convert Google Patents webpage text into well-formatted JSON while following these rules for accuracy and consistency.

JSON FORMATTING RULES:
1. Use ONLY double quotes for both keys and string values
2. Do not include trailing commas in arrays or objects
3. Ensure all keys are properly quoted
4. Provide ONLY a valid JSON object with no surrounding text
5. Use null or "Not explicitly stated" for missing information
6. Format multi-line text in clean, indented YAML style
7. Keep keys simple with NO colons within key names
8. Format "potential_limitations" as a simple string, not key-value pairs
9. Independent claims should never reference earlier claims
10. Include country code in patent numbers (e.g., "USXXXXXXXX")
11. Use full and accurate patent titles

{
  "metadata": {
    "extraction_date": "YYYY-MM-DD",
    "extraction_version": "1.0"
  },
  
  "bibliographic_information": {
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
    "legal_status": "Current status - active, pending, withdrawn, expired, etc."
  },
  
  "technical_content": {
    "key_technology": "Maximum 3-word technology description",
    "category_of_technology": [
      "Main technological field 1",
      "Main technological field 2",
      "Main technological field 3"
    ],
    "one_liner_summary": "One-sentence summary of the patent including the value proposition",
    "five_keypoints_summary": [
      "Key point 1",
      "Key point 2",
      "Key point 3",
      "Key point 4",
      "Key point 5"
    ],
    "abstract": [
      "Abstract paragraph 1",
      "Abstract paragraph 2"
    ],
    "background_summary": [
      "Background point 1",
      "Background point 2",
      "Background point 3"
    ],
    "key_components": [
      "Component 1",
      "Component 2",
      "Component 3"
    ],
    "technical_advantages": [
      "Advantage 1",
      "Advantage 2",
      "Advantage 3"
    ],
    "method_steps": [
      "Step 1",
      "Step 2",
      "Step 3"
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
    ]
  },
  
  "claim_analysis": {
    "independent_claims": [
      {
        "claim_number": 1,
        "summary": "Summary of the first independent claim"
      },
      {
        "claim_number": 11,
        "summary": "Summary of the second independent claim"
      }
    ],
    "claim_tree": {
      "1": ["2", "3", "4"],
      "11": ["12", "13", "14"]
    },
    "claim_terms_of_interest": [
      "Term 1",
      "Term 2",
      "Term 3"
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
    ]
  },
  
  "legal_assessment": {
    "prosecution_history_summary": [
      "Prosecution point 1",
      "Prosecution point 2"
    ],
    "claim_strength_assessment": [
      "Strength point 1",
      "Strength point 2",
      "Strength point 3"
    ],
    "invalidation_risks": [
      "Risk 1",
      "Risk 2"
    ],
    "litigation_history": [
      "Litigation item 1",
      "Litigation item 2"
    ],
    "potential_limitations": "Text describing limitations of the patent technology",
    "subject_quality": "good/average/bad rating based on claim scope, novelty, and commercial potential"
  },
  
  "commercial_assessment": {
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
      "User type 2",
      "User type 3"
    ],
    "value_proposition": [
      "Value point 1",
      "Value point 2",
      "Value point 3"
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
    "industry_adoption": [
      "Adoption example 1",
      "Adoption example 2"
    ],
    "licensing_information": [
      "Licensing info 1",
      "Licensing info 2"
    ],
    "estimated_market_value": "Estimated value range"
  },
  
  "citation_information": {
    "forward_citations_count": "Number as string",
    "backward_citations_count": "Number as string",
    "list_of_forward_citations": [
      "USXXXXXXX1",
      "EPXXXXXXX1",
      "USXXXXXXX2"
    ],
    "list_of_backward_citations": [
      "USXXXXXXX2",
      "JPXXXXXXX1",
      "USXXXXXXX3"
    ],
    "list_of_non_patent_citations": [
      "Citation 1",
      "Citation 2",
      "Citation 3"
    ],
    "citation_landscape": [
      "Landscape observation 1",
      "Landscape observation 2"
    ],
    "self_citations": [
      "USXXXXXXX3",
      "USXXXXXXX4"
    ],
    "most_cited_by": [
      "Company 1",
      "Company 2",
      "Company 3"
    ],
    "confidence_score": {
      "citation_analysis": "4",
      "market_assessment": "3"
    }
  },
  
  "patent_family_information": {
    "family_id": "INPADOC or Derwent family identifier",
    "us_family": {
      "patents": [
        "USXXXXXXX5",
        "USXXXXXXX6",
        "USXXXXXXX7"
      ],
      "applications": [
        "US15/XXX,XXX",
        "US16/XXX,XXX",
        "US17/XXX,XXX"
      ]
    },
    "ep_family": {
      "patents": [
        "EPXXXXXXX2",
        "EPXXXXXXX3"
      ],
      "applications": [
        "EPXXXXXXXX.XA",
        "EPXXXXXXXX.XA"
      ]
    },
    "jp_family": {
      "patents": [
        "JPXXXXXXX2",
        "JPXXXXXXX3"
      ],
      "applications": [
        "JPXXXXXXXXXA",
        "JPXXXXXXXXXA"
      ]
    },
    "cn_family": {
      "patents": [
        "CNXXXXXXXXX",
        "CNXXXXXXXXX"
      ],
      "applications": [
        "CNXXXXXXXXX.XA",
        "CNXXXXXXXXX.XA"
      ]
    },
    "wipo_family": {
      "publications": [
        "WOXXXXXXXX",
        "WOXXXXXXXY"
      ],
      "applications": [
        "PCT/XX/XXXX/XXXXX",
        "PCT/XX/XXXX/XXXXY"
      ]
    },
    "other_family": {
      "patents": [
        "KRXXXXXXXXXX",
        "KRXXXXXXXXXY"
      ],
      "applications": [
        "KRXXXXXXXXXXXA",
        "KRXXXXXXXXXXXB"
      ]
    },
    "priority_applications": [
      "USXXXXXXXX",
      "PCT/XX/XXXX/XXXXX"
    ],
    "related_applications": [
      "US15/XXX,XXX",
      "US16/XXX,XXX"
    ]
  },
  
  "classification_information": {
    "cpc_classifications": [
      "CPC code 1: Description 1",
      "CPC code 2: Description 2",
      "CPC code 3: Description 3"
    ],
    "ipc_classifications": [
      "IPC code 1: Description 1",
      "IPC code 2: Description 2"
    ],
    "uspc_classifications": [
      "USPC code 1: Description 1",
      "USPC code 2: Description 2"
    ],
    "related_technologies": [
      "Related technology 1",
      "Related technology 2",
      "Related technology 3"
    ]
  },
  
  "visual_information": {
    "drawings_count": "Number as string",
    "drawings_description": [
      "Figure 1: Description 1",
      "Figure 2: Description 2",
      "Figure 3: Description 3"
    ],
    "key_figures": [
      "Figure 1",
      "Figure 3",
      "Figure 5"
    ]
  }
}

KEY INSTRUCTIONS:
1. Include ALL relevant items in array fields, not just what's shown in the template
2. For citation lists, include ONLY patent/publication numbers (e.g., "USXXXXXXXX")
3. Base analytical fields (market impact, claim strength) on objective evidence from the patent
4. Format dates consistently as YYYY-MM-DD
5. Keep one-liner fields under 150 characters and summary points under 100 characters
6. For claim_tree, use claim numbers as strings and show dependencies correctly
7. For patent families, separate patents from applications in their respective arrays
8. Provide confidence scores on a 1-5 scale for subjective assessments
9. Use appropriate array sizes: 5 points for five_keypoints_summary, 3 points for most other arrays
10. For key_technology, provide a concise 1-3 word description of the core technology
11. For subject_quality, provide a single rating of "good", "average", or "bad" based on claim scope, novelty, and commercial potential

COMMON ERRORS TO AVOID:
- Unescaped special characters in strings
- Trailing commas at the end of arrays/objects
- Inconsistent date formats
- Missing quotes around keys or string values
- Improper formatting of nested objects
- Empty values (use null or "Not explicitly stated")
- Colons in key names
- Unbalanced brackets or braces
- Mixing data types (string vs. number)
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

    # Fix the common "potential_limitations": "value": "description" error pattern
    json_str = re.sub(
        r'"potential_limitations":\s*"[^"]*":\s*"([^"]*)"',
        r'"potential_limitations": "\1"',
        json_str,
    )

    # Fix any field with a format like "key": "value": "description"
    pattern = r'"([^"]+)":\s*"[^"]*":\s*"([^"]*)"'
    while re.search(pattern, json_str):
        json_str = re.sub(pattern, r'"\1": "\2"', json_str)

    # Fix unescaped quotes in string values
    # First, identify string values
    def fix_inner_quotes(match):
        value = match.group(1)
        # Replace unescaped quotes that aren't at the beginning/end
        if '"' in value[1:-1]:
            value = value[0] + value[1:-1].replace('"', '\\"') + value[-1]
        return value

    json_str = re.sub(r'("(?:[^"\\]|\\.)*")', fix_inner_quotes, json_str)

    return json_str


def parse_json_safely(json_str: str) -> dict[str, Any] | None:
    """Parse JSON with minimal processing - only repair if necessary.

    :param json_str: JSON string to parse
    :return: Parsed dictionary or None if parsing fails
    """
    # Remove markdown code block formatting if present
    json_str = re.sub(r"^```json\s*", "", json_str)
    json_str = re.sub(r"\s*```$", "", json_str)

    # Store original for debugging
    original_json_str = json_str
    
    # Create a log directory if it doesn't exist
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True, parents=True)
    
    # Create log file for parsing errors
    error_log = log_dir / f"json_parsing_errors_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

    with open(error_log, "a", encoding="utf-8") as log_file:
        # First attempt: try standard JSON parsing without any repair
        try:
            return json.loads(json_str)
        except json.JSONDecodeError as e:
            error_msg = f"Standard JSON parsing failed: {e}\nError at line {e.lineno}, column {e.colno}: {e.msg}"
            print(error_msg)
            
            # Log the error details
            log_file.write(f"=== JSON PARSING ERROR ===\n")
            log_file.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            log_file.write(f"Error: {e}\n")
            log_file.write(f"Line: {e.lineno}, Column: {e.colno}\n")
            log_file.write(f"Error message: {e.msg}\n\n")
            
            # Log the problematic context
            if 0 <= e.lineno - 1 < len(json_str.splitlines()):
                error_line = json_str.splitlines()[e.lineno - 1]
                log_file.write(f"Error line content: {error_line}\n")
                # Mark the error position
                pointer = ' ' * (e.colno - 1) + '^'
                log_file.write(f"Error position: {pointer}\n\n")
            
            # Try repair
            try:
                # Fall back to json_repair only if standard parsing fails
                log_file.write("Attempting repair with json_repair...\n")
                result = repair_json(json_str, return_objects=True, ensure_ascii=False)
                log_file.write("Repair successful!\n\n")
                return result
            except Exception as repair_e:
                error_msg = f"Error repairing JSON: {str(repair_e)}"
                print(error_msg)
                
                # Log repair error
                log_file.write(f"Repair failed: {repair_e}\n\n")
                
                # Save the problematic JSON for debugging
                debug_file = log_dir / f"problematic_json_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
                with open(debug_file, "w", encoding="utf-8") as f:
                    f.write(original_json_str)
                
                log_file.write(f"Full JSON saved to: {debug_file}\n")
                print(f"Saved problematic JSON to {debug_file}")
                
                # Return a dict with the raw response to preserve everything
                return {"raw_response": original_json_str}


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
        temperature=0.1,
        top_p=0.3,
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
    output_file = output_dir / f"{input_file.stem}.yaml"

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
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get all text files in the directory
    text_files = [
        f for f in input_path.iterdir() if f.is_file() and f.suffix.lower() in [".md"]
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
