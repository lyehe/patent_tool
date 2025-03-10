import os
import pandas as pd
from tqdm import tqdm
from datetime import datetime
import re
import concurrent.futures
from patent_extract import get_html, extract_data, save_data, PatentData, save_all_text


def clean_filename(name: str) -> str:
    """
    Create a safe filename from a string

    :param name: String to convert to filename
    :return: Safe filename string
    """
    return re.sub(r'[\\/*?:"<>|]', "", name.replace(" ", "_"))


def process_patent_url(
    args: tuple[str, str, str, int],
) -> tuple[PatentData | None, str | None]:
    """
    Process a single patent URL

    :param args: Tuple containing (url, output_format, output_path, timeout)
    :return: Tuple of (PatentData or None, error message or None)
    """
    url, output_format, output_path, timeout = args

    try:
        # Get HTML content
        html_content = get_html(url, is_url=True)

        if not html_content:
            return None, f"Failed to retrieve HTML content for {url}"

        # Extract patent data directly from HTML content
        patent_data = extract_data(html_content)

        if not patent_data or not patent_data.patent_number:
            return None, f"Failed to extract patent data from {url}"

        # Use only patent number for filename
        filename_base = patent_data.patent_number
        output_file = os.path.join(output_path, filename_base)
        
        # Save the raw text content
        text_dir = os.path.join(output_path, "text", "raw")
        os.makedirs(text_dir, exist_ok=True)
        raw_text_file = os.path.join(text_dir, f"{filename_base}.txt")
        save_all_text(html_content, raw_text_file)

        # Save patent data in requested format
        save_data(patent_data, output_file, output_format)

        return patent_data, None

    except Exception as e:
        import traceback

        trace = traceback.format_exc()
        return None, f"Error processing {url}: {str(e)}\n{trace}"


def extract_patents_from_csv(
    csv_file: str,
    output_format: str = "yaml",
    output_path: str = "output",
    limit: int | None = None,
    timeout: int = 30,
    max_workers: int = 10,
) -> list[PatentData]:
    """
    Extract patents from CSV file containing Google Patent URLs

    :param csv_file: Path to CSV file with patent links
    :param output_format: Output format ('yaml' or 'json')
    :param output_path: Directory to save individual patent files
    :param limit: Maximum number of patents to process
    :param timeout: Timeout in seconds for HTTP requests
    :param max_workers: Maximum number of concurrent workers
    :return: List of extracted PatentData objects
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_path, exist_ok=True)

    # Create a log file for errors
    log_path = os.path.join(output_path, "extraction_errors.log")

    # Read CSV file - skip the first row which contains the search URL
    df = pd.read_csv(csv_file, skiprows=1)

    # Apply limit if specified
    if limit is not None and limit > 0:
        df = df.head(limit)
        print(f"Limited to processing {limit} patents")

    # Check for URL column
    url_column = None
    for col in df.columns:
        if "url" in col.lower() or "link" in col.lower():
            url_column = col
            break

    if not url_column and "result link" in df.columns:
        url_column = "result link"

    if not url_column:
        raise ValueError(
            "CSV file must have a column containing URLs (with 'url' or 'link' in the name)"
        )

    # Filter out invalid URLs
    urls = [url for url in df[url_column].tolist() if url and not pd.isna(url)]

    # Process patents in parallel
    patents: list[PatentData] = []
    success_count = 0
    error_count = 0

    with open(log_path, "w", encoding="utf-8") as log_file:
        log_file.write(
            f"Patent Extraction Log - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        )
        log_file.write("=" * 80 + "\n\n")

        tasks = [(url, output_format, output_path, timeout) for url in urls]

        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            results = list(
                tqdm(
                    executor.map(process_patent_url, tasks),
                    total=len(tasks),
                    desc="Processing patents",
                )
            )

            for url, (patent_data, error_msg) in zip(urls, results):
                if error_msg:
                    log_file.write(f"ERROR - {url}: {error_msg}\n")
                    print(f"Error: {url}: {error_msg.split('\n')[0]}")
                    error_count += 1
                else:
                    patents.append(patent_data)
                    print(f"Saved: {patent_data.patent_number} - {patent_data.title}")
                    success_count += 1

        # Write summary
        summary = (
            f"\nExtraction Summary:\n"
            f"Total URLs: {len(urls)}\n"
            f"Successfully processed: {success_count}\n"
            f"Errors: {error_count}\n"
        )
        log_file.write(summary)
        print(summary)

    return patents


def main() -> int:
    """
    Main function to run batch patent extraction
    """
    import argparse

    parser = argparse.ArgumentParser(description="Batch process patents from CSV file")
    parser.add_argument("--csv", required=True, help="Input CSV file with patent URLs")
    parser.add_argument(
        "--format",
        choices=["yaml", "json"],
        default="yaml",
        help="Output format (default: yaml)",
    )
    parser.add_argument(
        "--output-dir", default="output", help="Output directory for YAML/JSON files"
    )
    parser.add_argument(
        "--limit", type=int, default=None, help="Limit the number of patents to process"
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=30,
        help="Timeout in seconds for HTTP requests (default: 30)",
    )
    parser.add_argument(
        "--retry",
        type=int,
        default=3,
        help="Number of retries for failed HTTP requests (default: 3)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=10,
        help="Maximum number of concurrent workers (default: 10)",
    )

    args = parser.parse_args()

    try:
        # Update request parameters and set the global timeout
        import requests

        requests.adapters.DEFAULT_RETRIES = args.retry
        # Set a global timeout for all requests
        import socket

        socket.setdefaulttimeout(args.timeout)

        # Extract patents
        patents = extract_patents_from_csv(
            args.csv,
            args.format,
            args.output_dir,
            limit=args.limit,
            timeout=args.timeout,
            max_workers=args.workers,
        )

        print(f"Processed {len(patents)} patents successfully")

    except Exception as e:
        print(f"Error: {e}")
        import traceback

        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    """
    Example usage:
    
    # Extract patents and save as YAML files:
    python batch_patent.py --csv patents.csv --format yaml --output-dir patent_data
    
    # Extract patents as JSON files:
    python batch_patent.py --csv patents.csv --format json --output-dir patent_data
    
    # Process only the first 10 patents with increased timeout:
    python batch_patent.py --csv patents.csv --limit 10 --timeout 60
    """
    exit(main())
