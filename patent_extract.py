import re
from dataclasses import dataclass, field
import requests
import aiohttp
import asyncio
from bs4 import BeautifulSoup
import os
from markdownify import markdownify as md
from pathlib import Path
import pandas as pd
from tqdm import tqdm
from tqdm.asyncio import tqdm_asyncio
from datetime import datetime
import concurrent.futures
import time


@dataclass
class PatentClaim:
    """Data class for patent claims with number, text and dependency information."""

    number: int
    text: str
    dependent_on: int | None = None


@dataclass
class PatentData:
    """Data class for storing basic patent information."""

    patent_number: str = ""
    title: str = ""
    assignees: list[str] = field(default_factory=list)
    inventors: list[str] = field(default_factory=list)
    priority_date: str = ""
    filing_date: str = ""
    publication_date: str = ""
    grant_date: str = ""
    abstract: str = ""
    description: str = ""
    claims: list[PatentClaim] = field(default_factory=list)


def clean_filename(name: str) -> str:
    """
    Create a safe filename from a string

    :param name: String to convert to filename
    :return: Safe filename string
    """
    return re.sub(r'[\\/*?:"<>|]', "", name.replace(" ", "_"))


def keep_only_ascii(text: str) -> str:
    """
    Remove any non-ASCII characters from text.

    Args:
        text: The text to clean

    Returns:
        Text containing only ASCII characters
    """
    if not text:
        return ""
    # Keep only ASCII characters (codes 0-127)
    return "".join(char for char in text if ord(char) < 128)


def get_html(input_source: str, is_url: bool, session=None) -> str:
    """
    Get HTML content from URL or file.

    Args:
        input_source: URL or file path
        is_url: Boolean indicating if input is a URL
        session: Optional requests session for connection pooling

    Returns:
        HTML content as string

    Raises:
        requests.exceptions.RequestException: For URL-related errors
        FileNotFoundError: If the input file doesn't exist
        IOError: For file read errors
    """
    try:
        if is_url:
            if session:
                response = session.get(input_source, timeout=30)
            else:
                headers = {
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
                    "Accept": "text/html,application/xhtml+xml,application/xml",
                    "Accept-Encoding": "gzip, deflate, br",
                    "Connection": "keep-alive",
                }
                response = requests.get(input_source, timeout=30, headers=headers)
            response.raise_for_status()
            return response.text
        else:
            with open(input_source, "r", encoding="utf-8") as f:
                return f.read()
    except (requests.exceptions.RequestException, FileNotFoundError, IOError) as e:
        raise e from None


def extract_text(soup: BeautifulSoup, itemprop: str) -> str:
    """
    Extract text from an element with the specified itemprop attribute.

    Args:
        soup: BeautifulSoup object containing the HTML
        itemprop: The itemprop attribute to search for

    Returns:
        The text content of the element or empty string if not found
    """
    element = soup.find(attrs={"itemprop": itemprop})
    if not element:
        return ""

    text = element.text.strip()

    # Remove "Abstract" from the beginning of abstract text
    if itemprop == "abstract" and text.startswith("Abstract"):
        text = text[len("Abstract") :].strip()

    return text


def extract_data(html_content: str) -> PatentData:
    """
    Extract basic patent metadata (simplified version).

    Args:
        html_content: HTML content of the patent document

    Returns:
        PatentData object containing basic extracted information
    """
    # Parse HTML
    soup = BeautifulSoup(html_content, "html.parser")

    data = PatentData()

    # Extract only the essential metadata for identification
    data.patent_number = extract_text(soup, "publicationNumber")
    data.title = extract_text(soup, "title")

    return data


def save_html_as_markdown(html_content: str, output_file: str) -> None:
    """
    Save HTML content as a Markdown file using markdownify with improved formatting.

    :param html_content: HTML content as a string
    :param output_file: Path to the output file
    """
    # Change file extension to .md if it's not already
    if not output_file.endswith(".md"):
        output_file = os.path.splitext(output_file)[0] + ".md"

    # Create the directory if it doesn't exist
    output_dir = os.path.dirname(output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # Create a BeautifulSoup object - use lxml for better performance
    soup = BeautifulSoup(html_content, "html.parser")

    # Remove script and style elements that contain non-visible text
    for element in soup(["script", "style", "noscript", "iframe"]):
        element.decompose()

    # Extract patent number and title before conversion for better heading
    data = extract_data(html_content)
    patent_number = data.patent_number
    title = data.title

    # Enhance tables before conversion
    tables = soup.find_all("table")
    for table in tables:
        # Add a class to indicate this is a table for post-processing
        table["class"] = table.get("class", []) + ["patent-table"]

    # Convert HTML to Markdown with improved settings
    markdown_text = md(
        str(soup), heading_style="ATX", strip=["img.patent-image-not-available"]
    )

    # Clean up markdown - more aggressive cleanup

    # Fix tables - make them more readable
    # Find markdown tables and improve their formatting
    table_pattern = (
        r"\|.*\|[\s]*\n\|[\s]*[-]+[\s]*\|[\s]*[-]+[\s]*\|.*\n(\|.*\|[\s]*\n)*"
    )

    def format_table(match):
        table_text = match.group(0)
        # Add extra newline before and after tables
        return "\n\n" + table_text + "\n\n"

    markdown_text = re.sub(
        table_pattern, format_table, markdown_text, flags=re.MULTILINE
    )

    # Improve citation links
    patent_link_pattern = (
        r"\[([A-Z]{2}\d+[A-Z0-9]*)\s+\((\w+)\)\]\(/patent/([A-Z0-9]+)/(\w+)\)"
    )
    markdown_text = re.sub(
        patent_link_pattern,
        r"[\1 (\2)](https://patents.google.com/patent/\3/\4)",
        markdown_text,
    )

    # Remove excessive newlines
    markdown_text = re.sub(r"\n{4,}", "\n\n\n", markdown_text)

    # Remove non-ASCII characters
    markdown_text = keep_only_ascii(markdown_text)

    # Remove common noise patterns
    markdown_text = re.sub(r"0\.000description\d+", "", markdown_text)
    markdown_text = re.sub(r"0\.000\w+\d+", "", markdown_text)

    # Improve section headers by adding horizontal rules
    section_headers = [
        "Abstract",
        "Claims",
        "Description",
        "Legal Events",
        "Classifications",
        "Citations",
    ]
    for header in section_headers:
        pattern = f"## {header}\\n"
        replacement = f"\n\n---\n\n## {header}\n\n"
        markdown_text = re.sub(pattern, replacement, markdown_text)

    # Write the markdown to file with better structure
    with open(output_file, "w", encoding="utf-8") as f:
        # Create a proper document header
        f.write(f"# Patent {patent_number}\n\n")

        if title:
            f.write(f"## {title}\n\n")

        # Add a table of contents section
        f.write("## Table of Contents\n\n")

        for header in section_headers:
            if header.lower() in markdown_text.lower():
                f.write(f"- [{header}](#{header.lower().replace(' ', '-')})\n")
        f.write("\n---\n\n")

        # Write the main content
        f.write(markdown_text)

    print(f"Formatted patent markdown saved to {output_file}")


def process_patent_url(
    args: tuple[str, Path, int, bool],
) -> tuple[PatentData | None, str | None]:
    """
    Process a single patent URL and save as markdown

    :param args: Tuple containing (url, output_path, timeout, force_reprocess)
    :return: Tuple of (PatentData or None, error message or None)
    """
    url, output_path, timeout, force_reprocess = args

    try:
        # Extract patent ID from URL for preliminary filename check
        patent_id = url.split("/patent/")[-1].split("/")[0]
        
        # Check if file already exists
        markdown_dir = output_path / "markdown"
        markdown_dir.mkdir(exist_ok=True, parents=True)
        md_file = markdown_dir / f"{patent_id}.md"
        
        if not force_reprocess and md_file.exists():
            print(f"Skipping {url} - file already exists: {md_file}")
            return PatentData(patent_number=patent_id), None

        # Get HTML content
        html_content = get_html(url, is_url=True)

        if not html_content:
            return None, f"Failed to retrieve HTML content for {url}"

        # Extract minimal patent data for reporting
        patent_data = extract_data(html_content)

        if not patent_data or not patent_data.patent_number:
            return None, f"Failed to extract patent data from {url}"

        # Use only patent number for filename
        filename_base = patent_data.patent_number

        # Save as markdown
        markdown_dir = output_path / "markdown"
        markdown_dir.mkdir(exist_ok=True, parents=True)
        md_file = markdown_dir / f"{filename_base}.md"
        save_html_as_markdown(html_content, str(md_file))

        return patent_data, None

    except Exception as e:
        import traceback

        trace = traceback.format_exc()
        return None, f"Error processing {url}: {str(e)}\n{trace}"


def extract_patents_from_csv(
    csv_file: str | Path,
    output_path: str | Path = "output",
    limit: int | None = None,
    timeout: int = 30,
    max_workers: int = 10,
    force_reprocess: bool = False,
) -> list[PatentData]:
    """
    Extract patents from CSV file containing Google Patent URLs

    :param csv_file: Path to CSV file with patent links
    :param output_path: Directory to save individual patent files
    :param limit: Maximum number of patents to process
    :param timeout: Timeout in seconds for HTTP requests
    :param max_workers: Maximum number of concurrent workers
    :param force_reprocess: Force reprocessing of patents even if files already exist
    :return: List of extracted PatentData objects
    """
    # Convert to Path objects
    csv_file = Path(csv_file)
    output_path = Path(output_path)

    # Create output directory if it doesn't exist
    output_path.mkdir(exist_ok=True, parents=True)

    # Create a log file for errors
    log_path = output_path / "extraction_errors.log"

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

        tasks = [(url, output_path, timeout, force_reprocess) for url in urls]

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


def extract_patents_from_txt(
    txt_file: str | Path,
    output_path: str | Path = "output",
    limit: int | None = None,
    timeout: int = 30,
    max_workers: int = 10,
    force_reprocess: bool = False,
) -> list[PatentData]:
    """
    Extract patents from a text file containing Google Patent URLs (one per line)

    :param txt_file: Path to text file with patent URLs
    :param output_path: Directory to save individual patent files
    :param limit: Maximum number of patents to process
    :param timeout: Timeout in seconds for HTTP requests
    :param max_workers: Maximum number of concurrent workers
    :param force_reprocess: Force reprocessing of patents even if files already exist
    :return: List of extracted PatentData objects
    """
    # Convert to Path objects
    txt_file = Path(txt_file)
    output_path = Path(output_path)

    # Create output directory if it doesn't exist
    output_path.mkdir(exist_ok=True, parents=True)

    # Create a log file for errors
    log_path = output_path / "extraction_errors.log"

    # Read text file line by line
    with open(txt_file, "r", encoding="utf-8") as f:
        urls = [line.strip() for line in f if line.strip()]

    # Apply limit if specified
    if limit is not None and limit > 0:
        urls = urls[:limit]
        print(f"Limited to processing {limit} patents")

    # Process patents in parallel
    patents: list[PatentData] = []
    success_count = 0
    error_count = 0

    with open(log_path, "w", encoding="utf-8") as log_file:
        log_file.write(
            f"Patent Extraction Log - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        )
        log_file.write("=" * 80 + "\n\n")

        tasks = [(url, output_path, timeout, force_reprocess) for url in urls]

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


async def process_patent_url_async(
    url: str, output_path: Path, session: aiohttp.ClientSession, force_reprocess: bool = False
) -> tuple[PatentData | None, str | None]:
    """
    Process a single patent URL asynchronously
    """
    try:
        # Extract patent ID from URL for preliminary filename check
        patent_id = url.split("/patent/")[-1].split("/")[0]
        
        # Check if file already exists
        markdown_dir = output_path / "markdown"
        markdown_dir.mkdir(exist_ok=True, parents=True)
        md_file = markdown_dir / f"{patent_id}.md"
        
        if not force_reprocess and md_file.exists():
            print(f"Skipping {url} - file already exists: {md_file}")
            return PatentData(patent_number=patent_id), None

        # Get HTML content asynchronously
        async with session.get(url, timeout=30) as response:
            if response.status != 200:
                return (
                    None,
                    f"Failed to retrieve HTML content for {url} (Status: {response.status})",
                )

            html_content = await response.text()

            # Extract minimal patent data for reporting
            patent_data = extract_data(html_content)

            if not patent_data or not patent_data.patent_number:
                return None, f"Failed to extract patent data from {url}"

            # Use only patent number for filename
            filename_base = patent_data.patent_number

            # Save as markdown
            markdown_dir = output_path / "markdown"
            markdown_dir.mkdir(exist_ok=True, parents=True)
            md_file = markdown_dir / f"{filename_base}.md"

            # Save HTML as markdown
            save_html_as_markdown(html_content, str(md_file))

            return patent_data, None

    except asyncio.TimeoutError:
        return None, f"Timeout while processing {url}"
    except Exception as e:
        import traceback

        trace = traceback.format_exc()
        return None, f"Error processing {url}: {str(e)}\n{trace}"


async def extract_patents_async(
    urls: list[str], output_path: Path, limit: int = None, concurrency: int = 10, force_reprocess: bool = False
) -> list[PatentData]:
    """
    Extract patents asynchronously with controlled concurrency
    """
    # Apply limit if specified
    if limit is not None and limit > 0:
        urls = urls[:limit]
        print(f"Limited to processing {limit} patents")

    # Create output directory and log file
    output_path.mkdir(exist_ok=True, parents=True)
    log_path = output_path / "extraction_errors.log"

    # Process patents concurrently with controlled concurrency
    patents: list[PatentData] = []
    success_count = 0
    error_count = 0

    # Custom TCP connector with optimized settings
    connector = aiohttp.TCPConnector(
        limit=concurrency, ttl_dns_cache=300, use_dns_cache=True, limit_per_host=5
    )

    # Create shared session for all requests
    async with aiohttp.ClientSession(
        connector=connector, raise_for_status=False
    ) as session:
        with open(log_path, "w", encoding="utf-8") as log_file:
            log_file.write(
                f"Patent Extraction Log - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
            )
            log_file.write("=" * 80 + "\n\n")

            # Create semaphore to control concurrency
            semaphore = asyncio.Semaphore(concurrency)

            async def fetch_with_semaphore(url):
                async with semaphore:
                    return await process_patent_url_async(url, output_path, session, force_reprocess)

            # Process URLs with progress bar
            tasks = [fetch_with_semaphore(url) for url in urls]
            results = await tqdm_asyncio.gather(*tasks, desc="Processing patents")

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


async def main_async():
    """
    Main function with performance monitoring and optimizations - async version
    """
    import argparse

    parser = argparse.ArgumentParser(
        description="Extract patent information and save as markdown"
    )

    # Create a mutually exclusive group for input file types
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--csv", help="Input CSV file with patent URLs")
    input_group.add_argument(
        "--txt", help="Input text file with patent URLs (one per line)"
    )
    input_group.add_argument("--url", help="Single patent URL to process")

    parser.add_argument(
        "--output-dir", default="output", help="Output directory for patent files"
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
    parser.add_argument(
        "--concurrency",
        type=int,
        default=10,
        help="Maximum concurrent requests (default: 10)",
    )
    parser.add_argument(
        "--sync",
        action="store_true",
        help="Use synchronous processing instead of async (slower but more reliable)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force reprocessing of patents even if files already exist",
    )

    args = parser.parse_args()

    start_time = time.time()

    try:
        # Update request parameters and set the global timeout
        requests.adapters.DEFAULT_RETRIES = args.retry
        # Set a global timeout for all requests
        import socket

        socket.setdefaulttimeout(args.timeout)

        # Convert string paths to Path objects
        output_dir = Path(args.output_dir)

        # Handle single URL case first
        if args.url:
            if args.sync:
                patent, error = process_patent_url((args.url, output_dir, args.timeout, args.force))
                if error:
                    print(f"Error: {error}")
                    return 1
                print(f"Processed patent: {patent.patent_number} - {patent.title}")
            else:
                # Process asynchronously
                connector = aiohttp.TCPConnector(limit=1, ttl_dns_cache=300)
                async with aiohttp.ClientSession(connector=connector) as session:
                    patent, error = await process_patent_url_async(
                        args.url, output_dir, session, args.force
                    )
                    if error:
                        print(f"Error: {error}")
                        return 1
                    print(f"Processed patent: {patent.patent_number} - {patent.title}")

        # Handle CSV input
        elif args.csv:
            if args.sync:
                # Use synchronous processing
                patents = extract_patents_from_csv(
                    args.csv,
                    output_dir,
                    limit=args.limit,
                    timeout=args.timeout,
                    max_workers=args.workers,
                    force_reprocess=args.force,
                )
            else:
                # Process asynchronously
                # Read CSV file - skip the first row
                df = pd.read_csv(args.csv, skiprows=1)

                # Find URL column
                url_column = next(
                    (
                        col
                        for col in df.columns
                        if "url" in col.lower() or "link" in col.lower()
                    ),
                    "result link",
                )

                if url_column not in df.columns:
                    raise ValueError("CSV file must have a column containing URLs")

                # Filter valid URLs
                urls = [
                    url for url in df[url_column].tolist() if url and not pd.isna(url)
                ]

                # Process asynchronously
                patents = await extract_patents_async(
                    urls, output_dir, limit=args.limit, concurrency=args.concurrency, force_reprocess=args.force
                )

        # Handle TXT input
        elif args.txt:
            if args.sync:
                # Use synchronous processing
                patents = extract_patents_from_txt(
                    args.txt,
                    output_dir,
                    limit=args.limit,
                    timeout=args.timeout,
                    max_workers=args.workers,
                    force_reprocess=args.force,
                )
            else:
                # Read text file
                with open(args.txt, "r", encoding="utf-8") as f:
                    urls = [line.strip() for line in f if line.strip()]

                # Process asynchronously
                patents = await extract_patents_async(
                    urls, output_dir, limit=args.limit, concurrency=args.concurrency, force_reprocess=args.force
                )

        # Calculate performance
        if args.url:
            patent_count = 1
        else:
            patent_count = len(patents)

        total_time = time.time() - start_time
        patents_per_second = patent_count / total_time if patent_count > 0 else 0

        print(f"Processed {patent_count} patents in {total_time:.2f} seconds")
        print(f"Average speed: {patents_per_second:.2f} patents/second")

    except Exception as e:
        print(f"Error: {e}")
        import traceback

        traceback.print_exc()
        return 1

    return 0


def main():
    """
    Entry point for the script - dispatches to async main
    """
    import sys

    # Check for Python 3.7+
    if sys.version_info < (3, 7):
        print("This script requires Python 3.7 or higher")
        return 1

    return asyncio.run(main_async())


if __name__ == "__main__":
    """
    Example usage:
    
    # Extract patents from a CSV file:
    python patent_extract.py --csv patents.csv --output-dir patent_data
    
    # Process only the first 10 patents with increased timeout:
    python patent_extract.py --csv patents.csv --limit 10 --timeout 60
    
    # Extract patents from a text file of URLs:
    python patent_extract.py --txt patent_urls.txt --output-dir patent_data
    
    # Process a single patent URL:
    python patent_extract.py --url https://patents.google.com/patent/US10000000 --output-dir patent_data
    
    # Use synchronous processing (slower but more reliable):
    python patent_extract.py --csv patents.csv --sync
    """
    import sys

    sys.exit(main())
