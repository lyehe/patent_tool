import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Optional, Dict, Any
from tqdm import tqdm
import sqlite3
from datetime import datetime
import re
from patent_extract import (
    get_html,
    extract_data,
    save_data,
    PatentData
)

def clean_filename(name: str) -> str:
    """
    Create a safe filename from a string
    
    :param name: String to convert to filename
    :return: Safe filename string
    """
    return re.sub(r'[\\/*?:"<>|]', "", name.replace(" ", "_"))

def extract_patents_from_csv(
    csv_file: str,
    output_format: str = "yaml",
    output_path: str = "output",
    db_path: Optional[str] = None,
    limit: Optional[int] = None,
    timeout: int = 30
) -> List[PatentData]:
    """
    Extract patents from CSV file containing Google Patent URLs
    
    :param csv_file: Path to CSV file with patent links
    :param output_format: Output format ('yaml', 'json', or 'sqlite')
    :param output_path: Directory to save individual patent files (for yaml/json)
    :param db_path: Path to SQLite database file (for sqlite format)
    :param limit: Maximum number of patents to process
    :param timeout: Timeout in seconds for HTTP requests
    :return: List of extracted PatentData objects
    """
    # Create output directory if it doesn't exist
    if output_format in ["yaml", "json"]:
        os.makedirs(output_path, exist_ok=True)
    
    # Create a log file for errors
    log_path = os.path.join(output_path if output_format in ["yaml", "json"] else os.path.dirname(db_path), 
                           "extraction_errors.log")
    
    # Read CSV file
    df = pd.read_csv(csv_file)
    
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
        raise ValueError("CSV file must have a column containing URLs (with 'url' or 'link' in the name)")
    
    # Process each patent URL
    patents = []
    success_count = 0
    error_count = 0
    
    with open(log_path, 'w', encoding='utf-8') as log_file:
        log_file.write(f"Patent Extraction Log - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        log_file.write("=" * 80 + "\n\n")
        
        for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing patents"):
            url = row[url_column]
            if not url or pd.isna(url):
                continue
            
            try:
                print(f"Processing: {url}")
                # Get HTML content
                html_content = get_html(url, is_url=True)
                
                if not html_content:
                    error_msg = f"Failed to retrieve HTML content for {url}"
                    print(error_msg)
                    log_file.write(f"ERROR - {url}: {error_msg}\n")
                    error_count += 1
                    continue
                
                # Extract patent data directly from HTML content
                patent_data = extract_data(html_content)
                
                # Validate patent data has essential fields
                if not patent_data.patent_number:
                    error_msg = f"No patent number extracted for {url}"
                    print(error_msg)
                    log_file.write(f"ERROR - {url}: {error_msg}\n")
                    error_count += 1
                    continue
                
                # Save in specified format
                if output_format in ["json", "yaml", "sqlite"]:
                    if output_format in ["json", "yaml"]:
                        filename = f"{clean_filename(patent_data.patent_number)}.{output_format}"
                        filepath = os.path.join(output_path, filename)
                    else:
                        filepath = db_path
                    
                    save_data(patent_data, filepath, output_format)
                    print(f"Saved: {patent_data.patent_number} - {patent_data.title}")
                
                patents.append(patent_data)
                success_count += 1
                
            except Exception as e:
                import traceback
                error_msg = f"Error processing {url}: {str(e)}"
                trace = traceback.format_exc()
                print(error_msg)
                log_file.write(f"ERROR - {url}: {error_msg}\n{trace}\n\n")
                error_count += 1
        
        # Write summary
        summary = f"\nExtraction Summary:\n" \
                  f"Total URLs: {len(df)}\n" \
                  f"Successfully processed: {success_count}\n" \
                  f"Errors: {error_count}\n"
        log_file.write(summary)
        print(summary)
    
    return patents

def parse_date(date_str: str) -> Optional[datetime]:
    """
    Parse date string into datetime object
    
    :param date_str: Date string to parse
    :return: Datetime object or None if parsing fails
    """
    if not date_str or pd.isna(date_str):
        return None
    
    date_formats = [
        "%Y-%m-%d",
        "%Y/%m/%d",
        "%d-%m-%Y",
        "%d/%m/%Y",
    ]
    
    for fmt in date_formats:
        try:
            return datetime.strptime(date_str, fmt)
        except ValueError:
            continue
    
    return None

def analyze_patents(csv_file: str, output_dir: str = "analytics") -> Dict[str, Any]:
    """
    Analyze patent data from CSV and generate analytics
    
    :param csv_file: Path to CSV file with patent data
    :param output_dir: Directory to save analytics output
    :return: Dictionary with analytics results
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Read CSV file
    df = pd.read_csv(csv_file)
    
    # Clean data
    for col in df.columns:
        if df[col].dtype == object:  # String columns
            df[col] = df[col].fillna("")
    
    # Analyze assignees distribution
    assignee_col = None
    for col in df.columns:
        if "assignee" in col.lower():
            assignee_col = col
            break
    
    # Identify columns
    inventor_col = next((col for col in df.columns if "inventor" in col.lower()), None)
    priority_date_col = next((col for col in df.columns if "priority" in col.lower() and "date" in col.lower()), None)
    filing_date_col = next((col for col in df.columns if "filing" in col.lower() or "creation" in col.lower()), None)
    publication_date_col = next((col for col in df.columns if "publication" in col.lower()), None)
    
    results = {}
    
    # 1. Assignees Distribution
    if assignee_col:
        # Some assignees might be separated by commas
        assignees = []
        for a in df[assignee_col]:
            if isinstance(a, str):
                parts = [p.strip() for p in a.split(',')]
                assignees.extend([p for p in parts if p])
        
        assignee_counts = pd.Series(assignees).value_counts()
        results['top_assignees'] = assignee_counts.head(10).to_dict()
        
        # Create plot
        plt.figure(figsize=(12, 6))
        sns.barplot(x=assignee_counts.head(10).index, y=assignee_counts.head(10).values)
        plt.xticks(rotation=45, ha='right')
        plt.title('Top 10 Patent Assignees')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'top_assignees.svg'), format='svg')
        plt.close()
    
    # 2. Inventors Distribution
    if inventor_col:
        inventors = []
        for i in df[inventor_col]:
            if isinstance(i, str):
                parts = [p.strip() for p in i.split(',')]
                inventors.extend([p for p in parts if p])
        
        inventor_counts = pd.Series(inventors).value_counts()
        results['top_inventors'] = inventor_counts.head(10).to_dict()
        
        # Create plot
        plt.figure(figsize=(12, 6))
        sns.barplot(x=inventor_counts.head(10).index, y=inventor_counts.head(10).values)
        plt.xticks(rotation=45, ha='right')
        plt.title('Top 10 Inventors')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'top_inventors.svg'), format='svg')
        plt.close()
    
    # 3. Time-based analysis
    if priority_date_col:
        # Parse dates
        df['priority_date_parsed'] = df[priority_date_col].apply(parse_date)
        valid_dates = [d for d in df['priority_date_parsed'] if d is not None]
        
        if valid_dates:
            date_years = pd.Series([d.year for d in valid_dates])
            year_counts = date_years.value_counts().sort_index()
            results['patents_by_year'] = year_counts.to_dict()
            
            # Create plot
            plt.figure(figsize=(12, 6))
            sns.lineplot(x=year_counts.index, y=year_counts.values)
            plt.title('Patents by Priority Year')
            plt.xlabel('Year')
            plt.ylabel('Number of Patents')
            plt.grid(True)
            plt.savefig(os.path.join(output_dir, 'patents_by_year.svg'), format='svg')
            plt.close()
    
    # Save results to YAML
    import yaml
    with open(os.path.join(output_dir, 'analytics_results.yaml'), 'w') as f:
        yaml.dump(results, f, default_flow_style=False)
    
    return results

def main() -> int:
    """
    Main function to run batch patent extraction and analysis
    """
    import argparse
    
    parser = argparse.ArgumentParser(description="Batch process patents from CSV file")
    parser.add_argument("--csv", required=True, help="Input CSV file with patent URLs")
    parser.add_argument("--format", choices=["yaml", "json", "sqlite"], default="yaml",
                        help="Output format (default: yaml)")
    parser.add_argument("--output-dir", default="output", 
                        help="Output directory for YAML/JSON files")
    parser.add_argument("--db-path", default="patents.db",
                        help="SQLite database path (for sqlite format)")
    parser.add_argument("--analytics", action="store_true", 
                        help="Generate analytics from the CSV data")
    parser.add_argument("--analytics-dir", default="analytics",
                        help="Directory for analytics output")
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit the number of patents to process")
    parser.add_argument("--timeout", type=int, default=30,
                        help="Timeout in seconds for HTTP requests (default: 30)")
    parser.add_argument("--retry", type=int, default=3,
                        help="Number of retries for failed HTTP requests (default: 3)")
    
    args = parser.parse_args()
    
    try:
        # Update request parameters if needed
        import requests
        requests.adapters.DEFAULT_RETRIES = args.retry
        
        # Extract patents
        patents = extract_patents_from_csv(
            args.csv, 
            args.format, 
            args.output_dir, 
            args.db_path,
            limit=args.limit,
            timeout=args.timeout
        )
        
        print(f"Processed {len(patents)} patents successfully")
        
        # Generate analytics if requested
        if args.analytics:
            results = analyze_patents(args.csv, args.analytics_dir)
            print(f"Analytics saved to {args.analytics_dir}")
        
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
    
    # Extract patents to SQLite database and generate analytics:
    python batch_patent.py --csv patents.csv --format sqlite --db-path patents.db --analytics
    
    # Process only the first 10 patents with increased timeout:
    python batch_patent.py --csv patents.csv --limit 10 --timeout 60
    """
    exit(main())