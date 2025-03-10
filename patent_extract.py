import re
import json
from dataclasses import dataclass, field

import yaml
import requests
from bs4 import BeautifulSoup, Tag
import os


@dataclass
class PatentClaim:
    """Data class for patent claims with number, text and dependency information."""
    number: int
    text: str
    dependent_on: int | None = None


@dataclass
class PatentData:
    """Data class for storing all patent information."""

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

    def to_dict(self) -> dict[str, object]:
        """Convert patent data to dictionary for serialization."""
        return {
            "patent_number": self.patent_number,
            "title": self.title,
            "assignees": self.assignees,
            "inventors": self.inventors,
            "priority_date": self.priority_date,
            "filing_date": self.filing_date,
            "publication_date": self.publication_date,
            "grant_date": self.grant_date,
            "abstract": self.abstract,
            "description": self.description,
            "claims": [claim.__dict__ for claim in self.claims],
        }


def clean_foreign_languages(text: str) -> str:
    """
    Remove foreign language text and keep only the English translation.

    Args:
        text: The text to process

    Returns:
        Text with only English content
    """
    if not text:
        return ""

    # Common markers that indicate an English translation follows
    translation_markers = [
        r"ENGLISH TRANSLATION:",
        r"TRANSLATION:",
        r"\(ENGLISH\)",
        r"English Translation:",
        r"English translation:",
        r"In English:",
        r"Translation to English:",
    ]

    # Check if any of the markers are present
    for marker in translation_markers:
        if re.search(marker, text, re.IGNORECASE):
            # Keep only the part after the marker
            parts = re.split(marker, text, flags=re.IGNORECASE, maxsplit=1)
            if len(parts) > 1:
                return parts[1].strip()

    # If no direct translation markers, check for language section headers
    language_headers = [
        # Look for the pattern: non-English section followed by an English section
        r"(?:\n|\r\n|\r)([A-Z\s]+)(?:\n|\r\n|\r).*?(?:\n|\r\n|\r)ENGLISH(?:\n|\r\n|\r)",
        # Two column format headers
        r"(?:\n|\r\n|\r)([A-Z\s]+)\s+\|\s+ENGLISH(?:\n|\r\n|\r)",
    ]

    for pattern in language_headers:
        match = re.search(pattern, text, re.DOTALL)
        if match:
            # Identified a non-English language section followed by English
            foreign_lang = match.group(1).strip()
            # Try to extract just the English portion
            # Split by the language header + English header pattern
            sections = re.split(
                f"{foreign_lang}.*?ENGLISH", text, flags=re.DOTALL | re.IGNORECASE
            )
            if len(sections) > 1:
                return sections[-1].strip()

    # WO patent specific: Look for sections with bracketed language indicators
    bracketed_lang_pattern = r"\[(DE|FR|ES|IT|JP|RU|CN|KR)?\].*?\[(EN)\]"
    if re.search(bracketed_lang_pattern, text, re.DOTALL | re.IGNORECASE):
        # Find all EN sections and concatenate them
        en_sections = []
        for match in re.finditer(
            r"\[(EN)\](.*?)(?=\[[A-Z]{2}\]|\Z)", text, re.DOTALL | re.IGNORECASE
        ):
            en_sections.append(match.group(2).strip())
        if en_sections:
            return "\n\n".join(en_sections)

    # If no translation markers found, return the original text
    # This assumes the text is already in English or there's no clear translation
    return text


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


def get_html(input_source: str, is_url: bool) -> str:
    """
    Get HTML content from URL or file.

    Args:
        input_source: URL or file path
        is_url: Boolean indicating if input is a URL

    Returns:
        HTML content as string

    Raises:
        requests.exceptions.RequestException: For URL-related errors
        FileNotFoundError: If the input file doesn't exist
        IOError: For file read errors
    """
    try:
        if is_url:
            response = requests.get(input_source, timeout=30)
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


def extract_list(soup: BeautifulSoup, prop: str) -> list[str]:
    """
    Extract a list of text values from elements with specified itemprop.

    Args:
        soup: BeautifulSoup object containing the HTML
        prop: The itemprop attribute to search for

    Returns:
        List of text values from matching elements
    """
    return [
        item.strip()
        for item in [
            elem.text.strip() for elem in soup.find_all(attrs={"itemprop": prop})
        ]
        if item
    ]


def extract_date(dt_element) -> str:
    """
    Extract date from a definition term element.

    Args:
        dt_element: BeautifulSoup dt element potentially containing date

    Returns:
        Extracted date string or empty string if not found
    """
    # First try to find a time element
    time_elem = dt_element.find_next("time")
    if time_elem and time_elem.get("datetime"):
        return time_elem.get("datetime")

    # Next try to find a definition description element
    dd = dt_element.find_next("dd")
    if dd:
        time_in_dd = dd.find("time")
        if time_in_dd and time_in_dd.get("datetime"):
            return time_in_dd.get("datetime")
        return dd.text.strip()

    # Last resort: check sibling text
    sibling = dt_element.find_next_sibling()
    return sibling.text.strip() if sibling else ""


def extract_grant_date(soup: BeautifulSoup) -> str:
    """
    Extract the grant date from patent events.

    Args:
        soup: BeautifulSoup object containing the HTML

    Returns:
        Grant date string or empty string if not found
    """
    event_elems = soup.find_all("dd", attrs={"itemprop": "events"})
    for event in event_elems:
        event_type = event.find("span", attrs={"itemprop": "title"})
        if event_type and "Application granted" in event_type.text:
            time_elem = event.find("time")
            if time_elem and time_elem.get("datetime"):
                return time_elem.get("datetime")
    return ""


def extract_assignees(soup: BeautifulSoup) -> list[str]:
    """
    Extract patent assignees, preserving all original names including foreign language content.

    Args:
        soup: BeautifulSoup object containing the HTML

    Returns:
        List of assignee names with original text preserved
    """
    assignees = []

    # Method 1: Look for elements with itemprop="assignee"
    assignee_elements = soup.find_all(attrs={"itemprop": "assignee"})
    for elem in assignee_elements:
        # Try to find the name within the element
        name_elem = elem.find(attrs={"itemprop": "name"})
        if name_elem:
            name = name_elem.get_text(strip=True)
        else:
            name = elem.get_text(strip=True)

        if name and name not in assignees:
            assignees.append(name)

    # Method 2: Look for assignee section by ID
    if not assignees:
        assignee_section = soup.find(id="assigneeSection") or soup.find(
            id="patent-assignees"
        )
        if assignee_section:
            # Look for list items or spans within the section
            items = assignee_section.find_all(["li", "span", "div", "a"])
            for item in items:
                name = item.get_text(strip=True)
                if (
                    name
                    and name not in assignees
                    and not any(c in name for c in ["Assignee", "Current", "Original"])
                ):
                    assignees.append(name)

    # Method 3: Look for dt/dd pairs with 'Assignee' label
    if not assignees:
        dt_elements = soup.find_all("dt")
        for dt in dt_elements:
            if "assignee" in dt.get_text().lower():
                # Find the next dd element, but handle case where it might not exist
                dd = dt.find_next("dd")
                # Only proceed if dd exists
                if dd is not None:  # This is the bug fix
                    name = dd.get_text(strip=True)
                    if name and name not in assignees:
                        assignees.append(name)

    # Method 4: Look for table rows with assignee information
    if not assignees:
        rows = soup.find_all("tr")
        for row in rows:
            cells = row.find_all("td")
            if len(cells) >= 2:
                header = cells[0].get_text().lower()
                if "assignee" in header:
                    value = cells[1].get_text(strip=True)
                    if value and value not in assignees:
                        assignees.append(value)

    # Method 5: For Google Patents, look for specific div classes
    if not assignees:
        for div in soup.find_all(
            "div",
            class_=lambda c: c and ("patent-assignee" in c or "assignee-list" in c),
        ):
            for item in div.find_all(["a", "span", "div"]):
                name = item.get_text(strip=True)
                if name and name not in assignees and len(name) > 1:
                    assignees.append(name)

    # Clean up results
    cleaned_assignees = []
    for assignee in assignees:
        # Remove any labels or common formatting issues
        cleaned = re.sub(r"^(Current|Original)\s+Assignee:?\s*", "", assignee)
        cleaned = cleaned.strip()
        if cleaned and cleaned not in cleaned_assignees:
            cleaned_assignees.append(cleaned)

    return cleaned_assignees


def extract_description(soup: BeautifulSoup) -> str:
    """
    Extract patent description with foreign language removal (optimized).
    
    Args:
        soup: BeautifulSoup object containing the HTML
        
    Returns:
        Description text with foreign language removed but English preserved
    """
    # Find the description section - try all selectors at once
    description_section = (
        soup.find("section", attrs={"itemprop": "description"}) or
        soup.find(id="descriptionText") or
        soup.find("section", id="description")
    )
    
    if not description_section:
        return ""
    
    # Create a single copy to work with
    desc_copy = BeautifulSoup(str(description_section), "html.parser")
    
    # Remove all foreign language spans in one operation
    for src in desc_copy.find_all("span", class_="google-src-text"):
        src.extract()
    
    # Get the text content with a single pass
    description_text = desc_copy.get_text(separator=" ", strip=True)
    
    # Clean up formatting with pre-compiled regex
    description_text = re.sub(r'\s{2,}', ' ', description_text)
    description_text = re.sub(r'\n{3,}', '\n\n', description_text)
    
    return description_text.strip()


def extract_claims(claims_section) -> list[PatentClaim]:
    """
    Extract patent claims with optimized performance, preserving complex chemical formulas.
    
    :param claims_section: The claims section of the patent document
    :return: List of PatentClaim objects
    """
    if not claims_section:
        return []
    
    # Pre-compile regex patterns
    claim_num_pattern = re.compile(r'claim-?(\d+)|c(\d+)$|CLM-0*(\d+)|en-cl(\d+)|num="?(\d+)"?')
    dependency_pattern = re.compile(r'(?:according to|as claimed in|of|in|as in|per)\s+claim\s+(\d+)', re.IGNORECASE)
    claim_pattern = re.compile(r'^\s*(\d+)[\.\)]')
    whitespace_pattern = re.compile(r'\s{2,}')
    
    # Create a copy to work with but keep original structure
    claims_copy = BeautifulSoup(str(claims_section), "html.parser")
    
    # Remove only the google-src-text spans (foreign language content)
    for src in claims_copy.find_all("span", class_="google-src-text"):
        src.extract()
    
    claims = []
    processed_claims = set()  # Track which claims we've already processed
    
    # Try multiple approaches to ensure we extract all claims
    
    # Method 1: Handle standard format with div.claim elements
    claim_divs = claims_copy.find_all('div', class_='claim')
    
    if claim_divs:
        for claim_div in claim_divs:
            # Extract claim ID to avoid duplicates
            claim_id = claim_div.get('id', '')
            if claim_id in processed_claims or claim_div.find_parent('div', class_='claim'):
                continue
                
            # Extract claim number
            claim_num_str = claim_div.get('num', '')
            if not claim_num_str and claim_id:
                match = claim_num_pattern.search(claim_id)
                if match:
                    # Get first non-None group
                    for group in match.groups():
                        if group:
                            claim_num_str = group
                            break
            
            # Try extracting from a bold element if no number found
            if not claim_num_str:
                first_text = claim_div.get_text().strip().split('.')[0]
                if first_text.isdigit():
                    claim_num_str = first_text
            
            try:
                claim_num = int(claim_num_str)
            except (ValueError, TypeError):
                # If we couldn't extract a valid number, skip this claim
                continue
                
            # Extract claim text while preserving structure
            # First, try to use claim-text divs if they exist
            claim_text_elements = claim_div.find_all('div', class_='claim-text')
            
            if claim_text_elements:
                # Join all claim-text parts with proper spacing
                claim_parts = []
                for element in claim_text_elements:
                    # Convert to string, then parse to keep structure but remove unwanted tags
                    element_clean = BeautifulSoup(str(element), "html.parser")
                    
                    # Handle patent-image spans
                    image_spans = element_clean.find_all("span", class_="patent-image-not-available")
                    for span in image_spans:
                        span.replace_with("[CHEMICAL FORMULA]")
                    
                    # Add the text with structure preserved
                    part_text = element_clean.get_text(strip=True)
                    if part_text:
                        claim_parts.append(part_text)
                
                claim_text = " ".join(claim_parts)
            else:
                # Fallback: extract text directly from the claim div
                claim_text = claim_div.get_text(strip=True)
            
            # Clean up text
            claim_text = whitespace_pattern.sub(' ', claim_text)
            if claim_text.startswith(f"{claim_num}. "):
                claim_text = claim_text[len(f"{claim_num}. "):]
            elif claim_text.startswith(f"{claim_num}."):
                claim_text = claim_text[len(f"{claim_num}."):]
            elif claim_text.startswith(f"{claim_num} "):
                claim_text = claim_text[len(f"{claim_num} "):]
            
            # Detect dependency
            dependent_on = None
            
            # Check for claim-ref elements first
            claim_refs = claim_div.find_all('claim-ref')
            if claim_refs:
                for ref in claim_refs:
                    ref_id = ref.get('idref', '')
                    if ref_id:
                        # Try to extract number from reference ID
                        match = claim_num_pattern.search(ref_id)
                        if match:
                            for group in match.groups():
                                if group:
                                    try:
                                        ref_num = int(group)
                                        if ref_num < claim_num:
                                            dependent_on = ref_num
                                            break
                                    except (ValueError, TypeError):
                                        pass
            
            # If no dependency found, check text for references
            if not dependent_on:
                match = dependency_pattern.search(claim_text)
                if match:
                    try:
                        dep_num = int(match.group(1))
                        if dep_num < claim_num:
                            dependent_on = dep_num
                    except (ValueError, TypeError):
                        pass
            
            # Check for dependency by parent div class
            if not dependent_on and claim_div.parent:
                parent_class = claim_div.parent.get('class', [])
                if 'claim-dependent' in parent_class:
                    # Try to find out which claim it depends on
                    match = dependency_pattern.search(claim_text)
                    if match:
                        try:
                            dep_num = int(match.group(1))
                            if dep_num < claim_num:
                                dependent_on = dep_num
                        except (ValueError, TypeError):
                            pass
                    # If still no dependency found but it's marked as dependent,
                    # it probably depends on the previous claim
                    if not dependent_on and claim_num > 1:
                        dependent_on = claim_num - 1
            
            # Add the claim
            if claim_text:
                claims.append(PatentClaim(number=claim_num, text=claim_text, dependent_on=dependent_on))
                processed_claims.add(claim_id if claim_id else str(claim_num))
    
    # Method 2: Handle claims within ordered lists (WO applications)
    if not claims:
        ol_elements = claims_copy.find_all('ol')
        if ol_elements:
            for ol in ol_elements:
                li_elements = ol.find_all('li')
                for i, li in enumerate(li_elements, 1):
                    # Extract claim text
                    claim_text = li.get_text(strip=True)
                    
                    # Try to find claim number
                    match = claim_pattern.search(claim_text)
                    claim_num = i  # Default to position in list
                    if match:
                        try:
                            claim_num = int(match.group(1))
                            # Remove the claim number from the beginning
                            claim_text = claim_text[match.end():].strip()
                        except (ValueError, TypeError):
                            pass
                    
                    # Check for dependency
                    dependent_on = None
                    match = dependency_pattern.search(claim_text)
                    if match:
                        try:
                            dep_num = int(match.group(1))
                            if dep_num < claim_num:
                                dependent_on = dep_num
                        except (ValueError, TypeError):
                            pass
                    
                    if claim_text:
                        claims.append(PatentClaim(number=claim_num, text=claim_text, dependent_on=dependent_on))
    
    # Method 3: Handle original Google Patents format with <claim> elements
    if not claims:
        claim_elements = claims_copy.find_all('claim')
        if claim_elements:
            for claim_elem in claim_elements:
                # Extract claim number
                claim_num_str = claim_elem.get('num', '')
                if not claim_num_str and 'id' in claim_elem.attrs:
                    match = claim_num_pattern.search(claim_elem['id'])
                    if match:
                        for group in match.groups():
                            if group:
                                claim_num_str = group
                                break
                
                try:
                    claim_num = int(claim_num_str)
                except (ValueError, TypeError):
                    continue
                
                # Extract text with structure preserved
                claim_text = claim_elem.get_text(strip=True)
                
                # Check for dependency
                dependent_on = None
                if 'depends' in claim_elem.attrs:
                    try:
                        dependent_on = int(claim_elem['depends'])
                    except (ValueError, TypeError):
                        pass
                
                if not dependent_on:
                    match = dependency_pattern.search(claim_text)
                    if match:
                        try:
                            dep_num = int(match.group(1))
                            if dep_num < claim_num:
                                dependent_on = dep_num
                        except (ValueError, TypeError):
                            pass
                
                if claim_text:
                    claims.append(PatentClaim(number=claim_num, text=claim_text, dependent_on=dependent_on))
    
    # Method 4: Last resort - try to parse from plain text
    if not claims:
        # Try to find claim-like structures directly
        text = claims_copy.get_text()
        claim_matches = re.finditer(r'(\d+)\.\s+(.*?)(?=(?:\d+)\.|$)', text, re.DOTALL)
        
        for match in claim_matches:
            try:
                claim_num = int(match.group(1))
                claim_text = match.group(2).strip()
                
                # Check for dependency
                dependent_on = None
                dep_match = dependency_pattern.search(claim_text)
                if dep_match:
                    try:
                        dep_num = int(dep_match.group(1))
                        if dep_num < claim_num:
                            dependent_on = dep_num
                    except (ValueError, TypeError):
                        pass
                
                if claim_text:
                    claims.append(PatentClaim(number=claim_num, text=claim_text, dependent_on=dependent_on))
            except (ValueError, TypeError):
                continue
    
    # Special handling for missing claim 1
    has_claim_1 = any(claim.number == 1 for claim in claims)
    if not has_claim_1 and claims and claims[0].number > 1:
        # Look specifically for claim 1
        claim1_div = claims_copy.find('div', id=lambda x: x and ('cl0001' in x or 'cl001' in x or 'cl01' in x or 'cl1' in x))
        if claim1_div:
            # Process this claim separately
            claim_text_elements = claim1_div.find_all('div', class_='claim-text')
            if claim_text_elements:
                claim_parts = []
                for element in claim_text_elements:
                    # Handle image spans
                    element_clean = BeautifulSoup(str(element), "html.parser")
                    image_spans = element_clean.find_all("span", class_="patent-image-not-available")
                    for span in image_spans:
                        span.replace_with("[CHEMICAL FORMULA]")
                    part_text = element_clean.get_text(strip=True)
                    if part_text:
                        claim_parts.append(part_text)
                claim_text = " ".join(claim_parts)
                claims.append(PatentClaim(number=1, text=claim_text, dependent_on=None))
    
    # Sort claims by number
    claims.sort(key=lambda c: c.number)
    
    return claims


def extract_inventors(soup: BeautifulSoup) -> list[str]:
    """
    Extract patent inventors, preserving all original names including foreign language content.

    Args:
        soup: BeautifulSoup object containing the HTML

    Returns:
        List of inventor names with original text preserved
    """
    inventors = []

    # Method 1: Look for elements with itemprop="inventor"
    inventor_elements = soup.find_all(attrs={"itemprop": "inventor"})
    for elem in inventor_elements:
        # Try to find the name within the element
        name_elem = elem.find(attrs={"itemprop": "name"})
        if name_elem:
            name = name_elem.get_text(strip=True)
        else:
            name = elem.get_text(strip=True)

        if name and name not in inventors:
            inventors.append(name)

    # Method 2: Look for inventor section by ID or class
    if not inventors:
        inventor_section = soup.find(id="inventorSection") or soup.find(
            id="patent-inventors"
        )
        if inventor_section:
            # Look for list items or spans within the section
            items = inventor_section.find_all(["li", "span", "div", "a"])
            for item in items:
                name = item.get_text(strip=True)
                if (
                    name
                    and name not in inventors
                    and not any(c in name for c in ["Inventor", "Current", "Original"])
                ):
                    inventors.append(name)

    # Method 3: Look for dt/dd pairs with 'Inventor' label
    if not inventors:
        dt_elements = soup.find_all("dt")
        for dt in dt_elements:
            if "inventor" in dt.get_text().lower():
                dd = dt.find_next("dd")
                # Only proceed if dd exists (fixing the same bug)
                if dd is not None:
                    name = dd.get_text(strip=True)
                    if name and name not in inventors:
                        inventors.append(name)

    # Method 4: Look for table rows with inventor information
    if not inventors:
        rows = soup.find_all("tr")
        for row in rows:
            cells = row.find_all("td")
            if len(cells) >= 2:
                header = cells[0].get_text().lower()
                if "inventor" in header:
                    value = cells[1].get_text(strip=True)
                    if value and value not in inventors:
                        inventors.append(value)

    # Clean up results (removing labels but preserving all name text)
    cleaned_inventors = []
    for inventor in inventors:
        # Remove any labels but keep all name characters
        cleaned = re.sub(r"^(Current|Original)\s+Inventor:?\s*", "", inventor)
        cleaned = cleaned.strip()
        if cleaned and cleaned not in cleaned_inventors:
            cleaned_inventors.append(cleaned)

    return cleaned_inventors


def extract_abstract(soup: BeautifulSoup) -> str:
    """
    Extract patent abstract with optimized performance.
    
    Args:
        soup: BeautifulSoup object containing the HTML
        
    Returns:
        Abstract text with foreign language removed
    """
    # Try all selectors at once
    abstract_section = (
        soup.find("abstract") or 
        soup.find(attrs={"itemprop": "abstract"}) or
        soup.find("section", id="abstract")
    )
    
    if not abstract_section:
        return ""
    
    # Create a single copy
    abstract_copy = BeautifulSoup(str(abstract_section), "html.parser")
    
    # Remove all foreign language spans in one pass
    for src in abstract_copy.find_all("span", class_="google-src-text"):
        src.extract()
    
    # Get text content in one operation
    abstract_text = abstract_copy.get_text(strip=True)
    
    # Clean up formatting
    abstract_text = re.sub(r'\s{2,}', ' ', abstract_text)
    
    return abstract_text.strip()


def extract_data(html_content: str) -> PatentData:
    """
    Optimized function to extract patent data from HTML content.

    Args:
        html_content: HTML content of the patent document

    Returns:
        PatentData object containing extracted information
    """
    # Parse HTML once
    soup = BeautifulSoup(html_content, "html.parser")

    data = PatentData()

    # Extract basic metadata
    data.patent_number = extract_text(soup, "publicationNumber")
    data.title = extract_text(soup, "title")

    # Extract inventors and assignees - preserving foreign language
    data.inventors = extract_inventors(soup)
    data.assignees = extract_assignees(soup)

    # Extract dates
    data.filing_date = extract_text(soup, "filingDate")
    data.publication_date = extract_text(soup, "publicationDate")
    data.priority_date = extract_text(soup, "priorityDate")
    data.grant_date = extract_grant_date(soup)

    # Extract content, removing foreign language
    data.abstract = extract_abstract(soup)
    data.description = extract_description(soup)

    # Extract claims section
    claims_section = soup.find(id="claims") or soup.find(
        "section", attrs={"itemprop": "claims"}
    )
    data.claims = extract_claims(claims_section)

    return data


def save_data(data: PatentData, output: str, format: str) -> None:
    """
    Save patent data to specified format.

    Args:
        data: PatentData object
        output: Output file path
        format: Output format ('json', 'yaml', 'sqlite', 'text')

    Raises:
        ValueError: If an invalid format is provided
        IOError: If there's an error writing to the output file
    """
    try:
        # Create format-specific directories
        if format == "yaml":
            output_dir = os.path.join(os.path.dirname(output), "yaml")
            os.makedirs(output_dir, exist_ok=True)
            output_file = os.path.join(output_dir, os.path.basename(output) + ".yaml")
            with open(output_file, "w", encoding="utf-8") as f:
                yaml.dump(data.to_dict(), f, sort_keys=False, allow_unicode=True)
        elif format == "json":
            with open(output, "w", encoding="utf-8") as f:
                json.dump(data.to_dict(), f, indent=2, ensure_ascii=False)
        elif format == "text":
            output_dir = os.path.join(os.path.dirname(output), "text")
            os.makedirs(output_dir, exist_ok=True)
            output_file = os.path.join(output_dir, os.path.basename(output) + ".txt")
            
            # Function to clean and filter text
            def clean_text(text: str) -> str:
                # Remove foreign characters
                text = keep_only_ascii(text)
                
                # Remove common repetitive patterns/tags
                text = re.sub(r'0\.000description\d+', '', text)
                text = re.sub(r'0\.000\w+\d+', '', text)
                
                # Important headers to preserve
                important_headers = [
                    "cited by", "citations", "patent citations", "references", 
                    "non-patent citations", "classification", "primary examiner",
                    "assistant examiner", "attorney", "agent", "representative", 
                    "application number", "priority", "family", "related", 
                    "legal events", "designated states", "international application"
                ]
                
                # Remove lines with minimal information
                lines = text.split('\n')
                filtered_lines = []
                
                for line in lines:
                    line = line.strip()
                    # Keep important headers regardless of length
                    if any(header in line.lower() for header in important_headers):
                        filtered_lines.append(line)
                        continue
                        
                    # For other content - skip empty lines or single character lines
                    if line and len(line) > 2:
                        filtered_lines.append(line)
                
                return '\n'.join(filtered_lines)
            
            # Create cleaned versions of text data with foreign characters removed
            abstract = clean_text(data.abstract)
            description = clean_text(data.description)
            inventors = [keep_only_ascii(inv) for inv in data.inventors if inv.strip()]
            assignees = [keep_only_ascii(asg) for asg in data.assignees if asg.strip()]
            
            # Filter claims
            filtered_claims = []
            for claim in data.claims:
                claim_text = clean_text(claim.text)
                if claim_text:
                    filtered_claims.append((claim.number, claim.dependent_on, claim_text))
            
            # Create outline entries
            outline_entries = []
            outline_entries.append(f"PATENT: {data.patent_number}")
            outline_entries.append(f"TITLE: {keep_only_ascii(data.title)}")
            
            if inventors:
                outline_entries.append("INVENTORS")
            if assignees:
                outline_entries.append("ASSIGNEES")
            if data.priority_date or data.filing_date or data.publication_date or data.grant_date:
                outline_entries.append("DATES")
            if abstract:
                outline_entries.append("ABSTRACT")
            if description:
                outline_entries.append("DESCRIPTION")
            if filtered_claims:
                outline_entries.append("CLAIMS")
                for number, dependent_on, _ in filtered_claims:
                    dependency = f" (depends on claim {dependent_on})" if dependent_on else ""
                    outline_entries.append(f"  Claim {number}{dependency}")
            
            # Write structured text representation of the patent data
            with open(output_file, "w", encoding="utf-8") as f:
                # Write document outline
                f.write("=" * 80 + "\n")
                f.write("PATENT DOCUMENT OUTLINE\n")
                f.write("=" * 80 + "\n\n")
                
                for entry in outline_entries:
                    f.write(f"{entry}\n")
                
                f.write("\n")
                f.write("-" * 80 + "\n\n")
                
                # Write full document with original structure
                f.write("=" * 80 + "\n")
                f.write("PATENT DOCUMENT WITH ORIGINAL STRUCTURE\n")
                f.write("=" * 80 + "\n\n")
                
                f.write("=" * 80 + "\n")
                f.write(f"PATENT: {data.patent_number}\n")
                f.write(f"TITLE: {keep_only_ascii(data.title)}\n")
                f.write("=" * 80 + "\n\n")
                
                # Only write sections that have content
                if inventors:
                    f.write("-" * 40 + "\n")
                    f.write("INVENTORS\n")
                    f.write("-" * 40 + "\n")
                    for inventor in inventors:
                        f.write(f"- {inventor}\n")
                    f.write("\n")
                
                if assignees:
                    f.write("-" * 40 + "\n")
                    f.write("ASSIGNEES\n")
                    f.write("-" * 40 + "\n")
                    for assignee in assignees:
                        f.write(f"- {assignee}\n")
                    f.write("\n")
                
                # Dates section only if there's at least one date
                if data.priority_date or data.filing_date or data.publication_date or data.grant_date:
                    f.write("-" * 40 + "\n")
                    f.write("DATES\n")
                    f.write("-" * 40 + "\n")
                    if data.priority_date:
                        f.write(f"Priority Date: {data.priority_date}\n")
                    if data.filing_date:
                        f.write(f"Filing Date: {data.filing_date}\n")
                    if data.publication_date:
                        f.write(f"Publication Date: {data.publication_date}\n")
                    if data.grant_date:
                        f.write(f"Grant Date: {data.grant_date}\n")
                    f.write("\n")
                
                if abstract:
                    f.write("-" * 40 + "\n")
                    f.write("ABSTRACT\n")
                    f.write("-" * 40 + "\n")
                    f.write(f"{abstract}\n\n")
                
                if description:
                    f.write("-" * 40 + "\n")
                    f.write("DESCRIPTION\n")
                    f.write("-" * 40 + "\n")
                    f.write(f"{description}\n\n")
                
                if filtered_claims:
                    f.write("-" * 40 + "\n")
                    f.write("CLAIMS\n")
                    f.write("-" * 40 + "\n")
                    for number, dependent_on, text in filtered_claims:
                        dependency = f" (depends on claim {dependent_on})" if dependent_on else ""
                        f.write(f"Claim {number}{dependency}:\n")
                        f.write(f"{text}\n\n")
            
            print(f"Patent text saved to {output_file}")
            return
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        print(f"Patent data saved to {output_file if format in ['yaml', 'text'] else output} in {format} format")
    except Exception as e:
        raise IOError(f"Error saving data: {e}") from e


def extract_all_text(html_content: str) -> str:
    """
    Extract all text from a webpage by removing all HTML tags.
    
    :param html_content: HTML content as a string
    :return: Plain text content with HTML tags removed
    """
    soup = BeautifulSoup(html_content, "html.parser")
    
    # Remove script and style elements that contain non-visible text
    for element in soup(["script", "style"]):
        element.decompose()
    
    # Get the text
    text = soup.get_text(separator=" ", strip=True)
    
    # Handle whitespace
    lines = (line.strip() for line in text.splitlines())
    text = "\n".join(line for line in lines if line)
    
    # Remove foreign characters
    text = keep_only_ascii(text)
    
    # Remove common tags and patterns
    text = re.sub(r'0\.000description\d+', '', text)
    text = re.sub(r'0\.000\w+\d+', '', text)
    
    return text


def save_all_text(html_content: str, output_file: str) -> None:
    """
    Save all text from a webpage to a file with HTML tags removed.
    
    :param html_content: HTML content as a string
    :param output_file: Path to the output file
    """
    soup = BeautifulSoup(html_content, "html.parser")
    
    # Remove script and style elements that contain non-visible text
    for element in soup(["script", "style"]):
        element.decompose()
    
    # Get headings with separators
    headings = soup.find_all(["h1", "h2", "h3", "h4", "h5", "h6"])
    paragraphs = soup.find_all("p")
    lists = soup.find_all(["ul", "ol"])
    
    # More thorough citation section detection
    citation_sections = []

    # 1. Use broader selectors for finding citation sections
    citation_selectors = [
        # IDs
        '#citedBy', '#patentCitations', '#citations', '#referenceCited', '#references',
        '#forwardCitations', '#backwardCitations', '#patentLiterature', '#nonPatentLiterature',
        # Classes
        '.citation', '.citations', '.patent-citation', '.cited-by', '.reference',
        '.references', '.patent-doc', '.citation-list', '.cited-patents',
        # General patent document sections
        '[id*="citation"]', '[id*="cited"]', '[id*="reference"]',
        '[class*="citation"]', '[class*="cited"]', '[class*="reference"]'
    ]

    for selector in citation_selectors:
        try:
            sections = soup.select(selector)
            citation_sections.extend(sections)
        except Exception:
            # Ignore any selector errors
            pass

    # 2. Broader heading text matching
    citation_heading_terms = [
        'citation', 'cited', 'reference', 'cited by', 'publication', 'patent document',
        'foreign', 'domestic', 'non-patent', 'u.s. patent', 'publication', 'document',
        'patent family', 'priority', 'also published'
    ]

    for heading in soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'dt', 'th', 'strong']):
        heading_text = heading.get_text().lower()
        if any(term in heading_text for term in citation_heading_terms):
            # Look at all siblings until we hit another heading
            current = heading.next_sibling
            while current and not (current.name in ['h1', 'h2', 'h3', 'h4', 'h5', 'h6']):
                if current.name:  # Only add tag elements, not NavigableString
                    citation_sections.append(current)
                current = current.next_sibling

    # 3. Include entire tables that might contain patent references
    for table in soup.find_all('table'):
        # Add any table with certain patterns in any cell
        cells = table.find_all(['td', 'th'])
        for cell in cells:
            cell_text = cell.get_text().lower()
            # Look for patent numbers or citation keywords
            if (re.search(r'[a-z]{1,3}\s*\d+', cell_text) or 
                any(term in cell_text for term in citation_heading_terms)):
                if table not in citation_sections:
                    citation_sections.append(table)
                    break

    # Function to clean text and filter out low-information content
    def clean_and_filter(text: str) -> str:
        # Remove foreign characters
        text = keep_only_ascii(text)
        
        # Remove common repetitive patterns/tags
        text = re.sub(r'0\.000description\d+', '', text)
        text = re.sub(r'0\.000\w+\d+', '', text)
        text = re.sub(r'Chemical compound', '', text)
        text = re.sub(r'Chemical group', '', text)
        text = re.sub(r'Natural products', '', text)
        text = re.sub(r'Substances', '', text)
        text = re.sub(r'Effects', '', text)
        text = re.sub(r'Methods', '', text)
        text = re.sub(r'Proteins', '', text)
        text = re.sub(r'Diseases', '', text)
        text = re.sub(r'Human genes', '', text)
        text = re.sub(r'Drugs', '', text)
        
        # Remove lines consisting only of a dash or other minimal content
        if text.strip() in ['-', '--', '---', '•', '·', '*', '**', '***']:
            return ''
            
        return text.strip()
    
    # Function to filter out lines with minimal information
    def filter_lines(text: str) -> str:
        """
        Filter lines with minimal filtering - preserving almost everything except definite noise.
        
        :param text: Text to filter
        :return: Filtered text with most content preserved
        """
        lines = text.split('\n')
        filtered_lines = []
        
        # Track when we're in a citation section to preserve all related content
        in_citation_section = False
        
        # Definite noise patterns to remove - ONLY these will be filtered out
        noise_patterns = [
            r'^-{5,}$',                         # Just dashes
            r'^\s*$',                           # Empty lines
            r'^[.,:;]{1,3}$',                   # Just punctuation
            r'^(\d+)$',                         # Just a single number
            r'^Copyright \d+',                  # Copyright notices
            r'^All rights reserved',            # Rights reserved
            r'^Terms of (Use|Service)',         # Terms mentions
            r'^Privacy Policy',                 # Privacy policy mentions
        ]
        
        # Citation section markers - when we detect these, preserve everything around them
        citation_markers = [
            'cited by', 'citations', 'patent citations', 'references', 'reference cited',
            'non-patent citations', 'forward citation', 'backward citation', 'citation list',
            'citing patents', 'cited patents', 'referenced by', 'references cited',
            'foreign patent', 'domestic patent', 'u.s. patent', 'publication',
            'other publications', 'patent literature', 'non-patent literature'
        ]
        
        # Patent number-like patterns
        patent_patterns = [
            r'[A-Z]{1,3}\s*\d+',                 # US 1234567
            r'[A-Z]{1,3}\d+[A-Z]\d*',            # US1234567A1
            r'\d{1,2}[/-]\d{1,2}[/-]\d{2,4}',    # Dates like MM/DD/YYYY
            r'\d{4}[/-]\d{1,2}[/-]\d{1,2}'       # Dates like YYYY/MM/DD
        ]
        
        for i, line in enumerate(lines):
            line = line.strip()
            
            # Skip truly empty lines
            if not line:
                continue
            
            # Always keep lines with patent numbers or dates
            if any(re.search(pattern, line) for pattern in patent_patterns):
                filtered_lines.append(line)
                in_citation_section = True
                continue
            
            # Check if this line contains a citation marker - if so, mark that we're in a citation section
            if any(marker in line.lower() for marker in citation_markers):
                filtered_lines.append(line)
                in_citation_section = True
                continue
            
            # If we're in a citation section, preserve more context (keep surrounding lines)
            if in_citation_section:
                # Reset citation section tracking only after several non-citation lines
                if len(line) < 3 or any(re.match(pattern, line) for pattern in noise_patterns):
                    # This might be the end of citation section, but don't reset yet
                    pass
                else:
                    # Keep this line as part of the citation section
                    filtered_lines.append(line)
                    continue
            
            # Only filter out lines that match specific noise patterns
            if any(re.match(pattern, line) for pattern in noise_patterns):
                continue
            
            # For anything else - KEEP IT by default
            filtered_lines.append(line)
        
        # Remove consecutive duplicate lines
        unique_lines = []
        prev_line = None
        for line in filtered_lines:
            if line != prev_line:
                unique_lines.append(line)
            prev_line = line
        
        return '\n'.join(unique_lines)
    
    # Extract all headings for the outline
    outline_headings = []
    for heading in headings:
        heading_level = int(heading.name[1])
        indent = "  " * (heading_level - 1)
        heading_text = clean_and_filter(heading.get_text(strip=True))
        if heading_text:  # Only include non-empty, filtered headings
            outline_headings.append(f"{indent}{'#' * heading_level} {heading_text}")
    
    # Process main content - keeping original structure
    processed_paragraphs = []
    for p in paragraphs:
        p_text = clean_and_filter(p.get_text(strip=True))
        if p_text and len(p_text) > 5:  # Only include substantial paragraphs
            processed_paragraphs.append(p_text)
    
    # Process lists - keeping original structure
    processed_lists = []
    for lst in lists:
        items = lst.find_all("li")
        list_items = []
        for item in items:
            item_text = clean_and_filter(item.get_text(strip=True))
            # Filter out chemical formulas and short items
            if (item_text and len(item_text) > 10 and 
                not re.search(r'[A-Z0-9]{10,}', item_text) and
                not re.search(r'\([A-Z]\)=', item_text)):
                list_items.append(f"- {item_text}")
        
        if list_items:
            processed_lists.append('\n'.join(list_items))
    
    # Extract citation content with better preservation
    processed_citations = []

    # Process citation sections and tables
    for section in citation_sections:
        # Try different extraction approaches for different element types
        if section.name == 'table':
            # For tables, preserve the structure with padding
            rows = section.find_all('tr')
            for row in rows:
                cells = row.find_all(['td', 'th'])
                if cells:
                    # Format as a table row
                    cell_texts = [cell.get_text().strip() for cell in cells]
                    formatted_row = " | ".join(text for text in cell_texts if text)
                    if formatted_row:
                        processed_citations.append(formatted_row)
        else:
            # For other elements, try to extract meaningful content
            # First check for list items
            list_items = section.find_all('li')
            if list_items:
                for item in list_items:
                    item_text = item.get_text().strip()
                    if item_text:
                        processed_citations.append(f"- {item_text}")
            else:
                # Next check for paragraphs or divs
                paragraphs = section.find_all(['p', 'div'])
                if paragraphs:
                    for p in paragraphs:
                        # Skip nested items we'll process separately
                        if p.find_parent(['li', 'td', 'th']) is not None:
                            continue
                        p_text = p.get_text().strip()
                        if p_text and len(p_text) > 3:
                            processed_citations.append(p_text)
                else:
                    # Fallback to direct text extraction
                    text = section.get_text().strip()
                    if text and len(text) > 3:
                        # Split by lines for better organization
                        lines = text.split('\n')
                        for line in lines:
                            line = line.strip()
                            if line and len(line) > 3:
                                processed_citations.append(line)

    # Get full text content
    raw_text = extract_all_text(html_content)
    filtered_text = filter_lines(raw_text)
    
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write("DOCUMENT OUTLINE\n")
        f.write("=" * 80 + "\n\n")
        
        # Write the outline of headings
        if outline_headings:
            f.write('\n'.join(outline_headings))
            f.write("\n\n")
            f.write("-" * 80 + "\n\n")
        
        f.write("=" * 80 + "\n")
        f.write("FULL DOCUMENT WITH ORIGINAL STRUCTURE\n")
        f.write("=" * 80 + "\n\n")
        
        # Original structure, with headers in place followed by content
        if headings:
            f.write("-" * 40 + "\n")
            f.write("HEADINGS\n")
            f.write("-" * 40 + "\n")
            for heading in headings:
                heading_level = int(heading.name[1])
                indent = "  " * (heading_level - 1)
                heading_text = clean_and_filter(heading.get_text(strip=True))
                if heading_text:  # Only write non-empty, filtered headings
                    f.write(f"{indent}{'#' * heading_level} {heading_text}\n")
            f.write("\n")
        
        if processed_paragraphs:
            f.write("-" * 40 + "\n")
            f.write("MAIN CONTENT\n")
            f.write("-" * 40 + "\n")
            for p_text in processed_paragraphs:
                f.write(f"{p_text}\n\n")
        
        if processed_lists:
            f.write("-" * 40 + "\n")
            f.write("LISTS\n")
            f.write("-" * 40 + "\n")
            for list_content in processed_lists:
                f.write(list_content)
                f.write("\n\n")
        
        if processed_citations:
            f.write("-" * 40 + "\n")
            f.write("CITATIONS AND REFERENCES\n")
            f.write("-" * 40 + "\n")
            for citation in processed_citations:
                f.write(f"{citation}\n")
            f.write("\n")
        
        if filtered_text:
            f.write("-" * 40 + "\n")
            f.write("FULL TEXT CONTENT\n")
            f.write("-" * 40 + "\n")
            f.write(filtered_text)
    
    print(f"Raw text saved to {output_file}")
