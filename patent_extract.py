import re
import json
from dataclasses import dataclass, field

import yaml
import requests
from bs4 import BeautifulSoup


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


def extract_names(soup: BeautifulSoup, entity_type: str) -> list[str]:
    """
    Extract inventors or assignees from patent HTML.

    :param soup: BeautifulSoup object containing the HTML
    :param entity_type: Type of entity to extract ('inventor' or 'assignee')
    :return: List of names with original text preserved
    """
    names = []

    # Method 1: Look for elements with itemprop attribute
    for elem in soup.find_all(attrs={"itemprop": entity_type}):
        if name_elem := elem.find(attrs={"itemprop": "name"}):
            name = name_elem.get_text(strip=True)
        else:
            name = elem.get_text(strip=True)
        if name and name not in names:
            names.append(name)

    # Method 2: Look for section by ID
    if not names:
        section_ids = [f"{entity_type}Section", f"patent-{entity_type}s"]
        section = next(
            (soup.find(id=id) for id in section_ids if soup.find(id=id)), None
        )
        if section:
            for item in section.find_all(["li", "span", "div", "a"]):
                name = item.get_text(strip=True)
                if (
                    name
                    and name not in names
                    and not any(
                        c in name
                        for c in [entity_type.capitalize(), "Current", "Original"]
                    )
                ):
                    names.append(name)

    # Method 3: Look for dt/dd pairs with label
    if not names:
        for dt in soup.find_all("dt"):
            if entity_type in dt.get_text().lower() and (dd := dt.find_next("dd")):
                name = dd.get_text(strip=True)
                if name and name not in names:
                    names.append(name)

    # Method 4: Look for table rows
    if not names:
        for row in soup.find_all("tr"):
            cells = row.find_all("td")
            if len(cells) >= 2 and entity_type in cells[0].get_text().lower():
                value = cells[1].get_text(strip=True)
                if value and value not in names:
                    names.append(value)

    # For assignees, try Google Patents specific classes
    if entity_type == "assignee" and not names:
        for div in soup.find_all(
            "div",
            class_=lambda c: c
            and (f"patent-{entity_type}" in c or f"{entity_type}-list" in c),
        ):
            for item in div.find_all(["a", "span", "div"]):
                name = item.get_text(strip=True)
                if name and name not in names and len(name) > 1:
                    names.append(name)

    # Clean up results
    pattern = rf"^(Current|Original)\s+{entity_type.capitalize()}:?\s*"
    return [
        re.sub(pattern, "", name).strip()
        for name in names
        if re.sub(pattern, "", name).strip()
    ]


def extract_section_text(
    soup: BeautifulSoup, selectors: list[tuple[str, str, tuple[str, str] | None]]
) -> str:
    """
    Extract text from a section using multiple selectors.

    :param soup: BeautifulSoup object
    :param selectors: List of selector tuples (selector_type, attribute, value)
    :return: Cleaned section text
    """
    section = None
    for selector_type, attr_name, attr_value in selectors:
        if selector_type == "tag":
            section = soup.find(attr_name, attrs={attr_value[0]: attr_value[1]})
        else:  # id
            section = soup.find(id=attr_name)
        if section:
            break

    if not section:
        return ""

    # Create a copy, remove foreign language spans
    section_copy = BeautifulSoup(str(section), "html.parser")
    for src in section_copy.find_all("span", class_="google-src-text"):
        src.extract()

    # Get text content and clean it
    content = section_copy.get_text(separator=" ", strip=True)
    content = re.sub(r"\s{2,}", " ", content)
    return content.strip()


def extract_claims(claims_section) -> list[PatentClaim]:
    """
    Extract patent claims with optimized performance, preserving complex chemical formulas.

    :param claims_section: The claims section of the patent document
    :return: List of PatentClaim objects
    """
    if not claims_section:
        return []

    # Prepare regex patterns and BS copy
    patterns = {
        "claim_num": re.compile(
            r'claim-?(\d+)|c(\d+)$|CLM-0*(\d+)|en-cl(\d+)|num="?(\d+)"?'
        ),
        "dependency": re.compile(
            r"(?:according to|as claimed in|of|in|as in|per)\s+claim\s+(\d+)",
            re.IGNORECASE,
        ),
        "claim_start": re.compile(r"^\s*(\d+)[\.\)]"),
        "whitespace": re.compile(r"\s{2,}"),
    }

    claims_copy = BeautifulSoup(str(claims_section), "html.parser")
    # Remove foreign language spans
    for src in claims_copy.find_all("span", class_="google-src-text"):
        src.extract()

    claims: list[PatentClaim] = []
    processed: set[str] = set()

    # Extract from claim div elements
    if claim_divs := claims_copy.find_all("div", class_="claim"):
        for div in claim_divs:
            claim_id = div.get("id", "")
            if claim_id in processed or div.find_parent("div", class_="claim"):
                continue

            # Get claim number from various sources
            claim_num_str = div.get("num", "")
            if not claim_num_str and claim_id:
                if match := patterns["claim_num"].search(claim_id):
                    claim_num_str = next((g for g in match.groups() if g), "")

            if not claim_num_str:
                first_text = div.get_text().strip().split(".")[0]
                if first_text.isdigit():
                    claim_num_str = first_text

            try:
                claim_num = int(claim_num_str)
            except (ValueError, TypeError):
                continue

            # Extract claim text
            if text_elements := div.find_all("div", class_="claim-text"):
                claim_parts = []
                for element in text_elements:
                    element_clean = BeautifulSoup(str(element), "html.parser")
                    for span in element_clean.find_all(
                        "span", class_="patent-image-not-available"
                    ):
                        span.replace_with("[CHEMICAL FORMULA]")
                    if part_text := element_clean.get_text(strip=True):
                        claim_parts.append(part_text)
                claim_text = " ".join(claim_parts)
            else:
                claim_text = div.get_text(strip=True)

            # Clean text
            claim_text = patterns["whitespace"].sub(" ", claim_text)
            prefixes = [f"{claim_num}. ", f"{claim_num}.", f"{claim_num} "]
            for prefix in prefixes:
                if claim_text.startswith(prefix):
                    claim_text = claim_text[len(prefix) :]
                    break

            # Detect dependency
            dependent_on = None

            # Check claim-ref elements
            for ref in div.find_all("claim-ref"):
                if ref_id := ref.get("idref", ""):
                    if match := patterns["claim_num"].search(ref_id):
                        if ref_num_str := next((g for g in match.groups() if g), None):
                            try:
                                ref_num = int(ref_num_str)
                                if ref_num < claim_num:
                                    dependent_on = ref_num
                                    break
                            except (ValueError, TypeError):
                                pass

            # Check text for references
            if not dependent_on:
                if match := patterns["dependency"].search(claim_text):
                    try:
                        dep_num = int(match.group(1))
                        if dep_num < claim_num:
                            dependent_on = dep_num
                    except (ValueError, TypeError):
                        pass

            # Check parent div class
            if (
                not dependent_on
                and div.parent
                and "claim-dependent" in div.parent.get("class", [])
            ):
                if match := patterns["dependency"].search(claim_text):
                    try:
                        dep_num = int(match.group(1))
                        if dep_num < claim_num:
                            dependent_on = dep_num
                    except (ValueError, TypeError):
                        pass
                # Default to previous claim
                if not dependent_on and claim_num > 1:
                    dependent_on = claim_num - 1

            if claim_text:
                claims.append(
                    PatentClaim(
                        number=claim_num, text=claim_text, dependent_on=dependent_on
                    )
                )
                processed.add(claim_id if claim_id else str(claim_num))

    # Try alternative formats if no claims found
    if not claims:
        # Method 2: Ordered lists
        if ol_elements := claims_copy.find_all("ol"):
            for ol in ol_elements:
                for i, li in enumerate(ol.find_all("li"), 1):
                    claim_text = li.get_text(strip=True)

                    # Get claim number
                    claim_num = i
                    if match := patterns["claim_start"].search(claim_text):
                        try:
                            claim_num = int(match.group(1))
                            claim_text = claim_text[match.end() :].strip()
                        except (ValueError, TypeError):
                            pass

                    # Check dependency
                    dependent_on = None
                    if match := patterns["dependency"].search(claim_text):
                        try:
                            dep_num = int(match.group(1))
                            if dep_num < claim_num:
                                dependent_on = dep_num
                        except (ValueError, TypeError):
                            pass

                    if claim_text:
                        claims.append(
                            PatentClaim(
                                number=claim_num,
                                text=claim_text,
                                dependent_on=dependent_on,
                            )
                        )

        # Method 3: <claim> elements
        elif claim_elements := claims_copy.find_all("claim"):
            for claim_elem in claim_elements:
                # Get claim number
                claim_num_str = claim_elem.get("num", "")
                if not claim_num_str and "id" in claim_elem.attrs:
                    if match := patterns["claim_num"].search(claim_elem["id"]):
                        claim_num_str = next((g for g in match.groups() if g), "")

                try:
                    claim_num = int(claim_num_str)
                except (ValueError, TypeError):
                    continue

                claim_text = claim_elem.get_text(strip=True)

                # Check dependency
                dependent_on = None
                if "depends" in claim_elem.attrs:
                    try:
                        dependent_on = int(claim_elem["depends"])
                    except (ValueError, TypeError):
                        pass

                if not dependent_on and (
                    match := patterns["dependency"].search(claim_text)
                ):
                    try:
                        dep_num = int(match.group(1))
                        if dep_num < claim_num:
                            dependent_on = dep_num
                    except (ValueError, TypeError):
                        pass

                if claim_text:
                    claims.append(
                        PatentClaim(
                            number=claim_num, text=claim_text, dependent_on=dependent_on
                        )
                    )

        # Method 4: Plain text parsing
        else:
            text = claims_copy.get_text()
            for match in re.finditer(
                r"(\d+)\.\s+(.*?)(?=(?:\d+)\.|$)", text, re.DOTALL
            ):
                try:
                    claim_num = int(match.group(1))
                    claim_text = match.group(2).strip()

                    dependent_on = None
                    if dep_match := patterns["dependency"].search(claim_text):
                        try:
                            dep_num = int(dep_match.group(1))
                            if dep_num < claim_num:
                                dependent_on = dep_num
                        except (ValueError, TypeError):
                            pass

                    if claim_text:
                        claims.append(
                            PatentClaim(
                                number=claim_num,
                                text=claim_text,
                                dependent_on=dependent_on,
                            )
                        )
                except (ValueError, TypeError):
                    continue

    # Check for missing claim 1
    if (
        claims
        and not any(claim.number == 1 for claim in claims)
        and claims[0].number > 1
    ):
        claim1_id_patterns = ["cl0001", "cl001", "cl01", "cl1"]
        if claim1_div := claims_copy.find(
            "div",
            id=lambda x: x and any(pattern in x for pattern in claim1_id_patterns),
        ):
            if text_elements := claim1_div.find_all("div", class_="claim-text"):
                claim_parts = []
                for element in text_elements:
                    element_clean = BeautifulSoup(str(element), "html.parser")
                    for span in element_clean.find_all(
                        "span", class_="patent-image-not-available"
                    ):
                        span.replace_with("[CHEMICAL FORMULA]")
                    if part_text := element_clean.get_text(strip=True):
                        claim_parts.append(part_text)
                claim_text = " ".join(claim_parts)
                claims.append(PatentClaim(number=1, text=claim_text, dependent_on=None))

    # Sort claims by number
    claims.sort(key=lambda c: c.number)
    return claims


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

    # Extract people and organizations
    data.inventors = extract_names(soup, "inventor")
    data.assignees = extract_names(soup, "assignee")

    # Extract dates
    data.filing_date = extract_text(soup, "filingDate")
    data.publication_date = extract_text(soup, "publicationDate")
    data.priority_date = extract_text(soup, "priorityDate")
    data.grant_date = extract_grant_date(soup)

    # Extract content sections
    data.abstract = extract_section_text(
        soup,
        [
            ("tag", "abstract", None),
            ("tag", "abstract", ("itemprop", "abstract")),
            ("id", "abstract", None),
        ],
    )

    data.description = extract_section_text(
        soup,
        [
            ("tag", "section", ("itemprop", "description")),
            ("id", "descriptionText", None),
            ("id", "description", None),
        ],
    )

    # Extract claims
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
        format: Output format ('json', 'yaml', 'text')

    Raises:
        ValueError: If an invalid format is provided
        IOError: If there's an error writing to the output file
    """
    try:
        if format == "json":
            with open(output, "w", encoding="utf-8") as f:
                json.dump(data.to_dict(), f, indent=2, ensure_ascii=False)
        elif format == "yaml":
            with open(output, "w", encoding="utf-8") as f:
                yaml.dump(data.to_dict(), f, sort_keys=False, allow_unicode=True)
        elif format == "text":
            print(data)
        else:
            raise ValueError(f"Unsupported format: {format}")
        print(f"Patent data saved to {output} in {format} format")
    except Exception as e:
        raise IOError(f"Error saving data: {e}") from e
