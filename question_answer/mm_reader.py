import re
import fitz
import pandas as pd
import json
import os
import difflib

# list of parameters to adjust if used for multiple document analysis
reading_box : tuple = (50, 60, 570, 720)
aim_length = 2000
min_length = 100

stop_patterns = [ 
    r'^.*\*.*$',                         # any line with ~ sign
    r'^.*\_.*$',                         # any line with _ sign
    r'^.*\=.*$',                         # any line with + sign
    r'^.*\~.*$',                         # any line with = sign

    r'^\d{2,}\s*$',                         # Multi-digit numbers
    r'^[a-z]{1,}\s*$',                    # Single lowercase word
    r'^[A-Z]{1,}\s*$',                    # Single uppercase word
    r'^[A-Z\s]+$',                      # Multiple capital words 

    r'^Figure\s*\d+.*$',                  # Figure captions
    r'^DETAIL\s*[A-Z0-9]+.*$',            # Detail labels
    r'^[^a-zA-Z0-9]{1,}$',                # Non-alphanumeric noise
    r'^(?:[A-Z][a-z]+(?:[\s\-\/]))+$',        # Multiple words with starting letter being a capital
    r'^(?=.*[A-Za-z])(?=.*\d)[A-Za-z\d]{4,}$',  # random part

    r'^\((?:(?!\b\d{1,2}\b)(?!\([a-z]\)$)[A-Za-z0-9\s]){3,}\)$',  # random code in bracket
    r'^\d+(?:\s*,\s*\d+)*,?$',           # list of numbers
    r'^\s*$\n'                           # blank lines
]

def prepare_doc(pdf_file, reading_box: tuple = None, stop_patterns: list[str] = None, aim_length: int = None, min_length: int = None):
    '''receive a input document and transform it into dictionaries with desired format'''
    # Use module-level defaults if not provided
    if reading_box is None:
        reading_box = (50, 60, 570, 720)
    if stop_patterns is None:
        stop_patterns = [
            r'^.*\*.*$',                         # any line with ~ sign
            r'^.*\_.*$',                         # any line with _ sign
            r'^.*\=.*$',                         # any line with + sign
            r'^.*\~.*$',                         # any line with = sign
            r'^\d{2,}\s*$',                         # Multi-digit numbers
            r'^[a-z]{1,}\s*$',                    # Single lowercase word
            r'^[A-Z]{1,}\s*$',                    # Single uppercase word
            r'^[A-Z\s]+$',                      # Multiple capital words 
            r'^Figure\s*\d+.*$',                  # Figure captions
            r'^DETAIL\s*[A-Z0-9]+.*$',            # Detail labels
            r'^[^a-zA-Z0-9]{1,}$',                # Non-alphanumeric noise
            r'^(?:[A-Z][a-z]+(?:[\s\-\/]))+$',        # Multiple words with starting letter being a capital
            r'^(?=.*[A-Za-z])(?=.*\d)[A-Za-z\d]{4,}$',  # random part
            r'^\((?:(?!\b\d{1,2}\b)(?!\([a-z]\)$)[A-Za-z0-9\s]){3,}\)$',  # random code in bracket
            r'^\d+(?:\s*,\s*\d+)*,?$',           # list of numbers
            r'^\s*$\n'                           # blank lines
        ]
    if aim_length is None:
        aim_length = 2000
    if min_length is None:
        min_length = 100
    doc = fitz.open(pdf_file)
    reading_area : fitz.Rect = fitz.Rect(reading_box)

    toc_hierarchy = process_toc(doc)
    answer_library = process_content(toc_hierarchy, doc, reading_area, stop_patterns, aim_length, min_length)

    df = pd.DataFrame(answer_library)
    
    # Remove rows where 'text' column has less than 30 characters
    df = df[df['text'].str.len() >= 30].reset_index(drop=True)
    
    # Remove duplicate rows based on 'text' column, keeping the first occurrence
    df = df.drop_duplicates(subset=['text'], keep='first').reset_index(drop=True)
    
    df.to_csv('question_answer/split_mm.csv', index=False)

    return df


def process_content(toc_hierarchy: list, doc: fitz.Document, reading_area: fitz.Rect, stop_patterns: list[str], aim_length: int, min_length: int):
    '''extract the useful sections of the document and cleans up the document'''
    extracted_answers: list[dict] = []
    for i, chapter in enumerate(toc_hierarchy):
        if i == len(toc_hierarchy) - 1:
            c_page_next = len(doc) - 1
            c_name_next = "END OF THE DOCUMENT"
        else:
            c_name, c_page = chapter['chapter'], chapter['page'] - 1
            c_name_next, c_page_next = toc_hierarchy[i+1]['chapter'], toc_hierarchy[i+1]['page'] - 1
        for j, title in enumerate(chapter["titles"]):
            if j == len(chapter["titles"]) - 1:
                t_page_next = c_page_next
                t_name_next = c_name_next
            else:
                t_name, t_page = title['title'], title['page'] - 1
                t_name_next, t_page_next = chapter["titles"][j+1]['title'], chapter["titles"][j+1]['page'] - 1
            for k, section in enumerate(title["sections"]):
                s_name, start_page = section['section'], section['page'] - 1

                if k == len(title["sections"]) -1:
                    end_page = t_page_next
                    s_name_next = t_name_next
                else:
                    s_name_next, s_page_next = title["sections"][k+1]['section'], title["sections"][k+1]['page'] - 1
                    if start_page == s_page_next:
                        end_page = None
                    else:
                        end_page = s_page_next + 1

                extracted_pages = extract_pages(start_page, end_page, doc)
                extracted_text = extract_text(extracted_pages, s_name, s_name_next, reading_area)
                specific_processing_1 = remove_documnet_specific_content_1(extracted_text)
                processed_text = process_text(specific_processing_1, stop_patterns)
                splited_text = split_text(processed_text, aim_length, min_length)

                if splited_text is  None:
                    continue
                else: 
                    for answer in splited_text:
                        extracted_answers.append({"chapter": c_name, "title": t_name, "text": answer})
    
    return extracted_answers


def split_text(texts: list[str], max_length, min_length):
    '''split the text into desired length and filter out too short sections'''
    answers = []
    if texts == None:
        return None
    else:
        lengths = [len(text) for text in texts]
        if sum(lengths) < min_length:
            return None
        else:
            cumulative_sum : int = 0
            cumulative_answer : str = ''
            for i, (words, length) in enumerate(zip(texts, lengths)):
                cumulative_sum += length
                cumulative_answer += words
                
                if cumulative_sum >= max_length:
                    cumulative_answer = re.sub("\n", " ", cumulative_answer)
                    answers.append(cumulative_answer)
                    cumulative_sum = 0
                    cumulative_answer = ''
                elif i == len(texts) - 1:
                    cumulative_answer = re.sub("\n", " ", cumulative_answer)
                    answers.append(cumulative_answer)
        
        return answers


def remove_documnet_specific_content_1 (extracted_text: str):
    '''remove tables -- however, I am thinking get rid of this function so it is more general'''
    start_pattern = r'(ITEM CODE|TASK)'
    end_pattern = r'\*\*\* End of Operation \d+ Inspection Items \*\*'

    start_match = re.search(start_pattern, extracted_text)
    end_match = re.search(end_pattern, extracted_text)

    # If both found
    if start_match and end_match:
        start_idx = start_match.start()
        end_idx = end_match.end()  # include the end marker in removal

        # Remove everything between them
        new_text = extracted_text[:start_idx] + extracted_text[end_idx:]
        return new_text
    else:
        return extracted_text


def process_text(extracted_text :  str, stop_patterns: list[str]):
    """ process the text into idea format"""
    # Check if the section header pattern exists
    header_pattern = r'(?m)(?=^[A-Z]\.)'
    if not re.search(header_pattern, extracted_text):
        return None

    # Split into sections (A., B., C., etc.)
    sections = re.split(header_pattern, extracted_text)  
    
    cleaned_sections : list[str] = []
    
    # Process each section
    for sec in sections:        
        if not sec.strip():  # skip empty splits
            continue 
        
        cleaned_section = clean_text(sec, stop_patterns)
        cleaned_sections.append(cleaned_section.strip())

    return cleaned_sections

def clean_text(text: str, stop_patterns: list[str]):
    text = re.sub(r'^\b(?:[A-Z]|\d+)\.\s?$', "", text, flags=re.MULTILINE)
    text = text.strip()

    lines = text.splitlines()
    space_counts = [line.count(" ") for line in lines]
    found = any(all(x <= 3 for x in space_counts[i:i+3]) for i in range(len(space_counts)-2))

    if found:
        cleaned_lines = [line for line, count in zip(lines, space_counts) if count > 4]
        
    else:
        cleaned_lines = lines  # Keep all lines if condition not met

    compact_lines = remove_repitition(cleaned_lines)
    
    cleaned_text = "\n".join(compact_lines)
    cleaned_text = re.sub(r'\((?![a-z0-9]\))[^)]*?\)', "", cleaned_text, flags=re.MULTILINE)

    for stop_pattern in stop_patterns:
        cleaned_text = re.sub(stop_pattern, "", cleaned_text, flags=re.MULTILINE)
    
    return cleaned_text


def remove_repitition(lines: list[str], threshold: float = 0.75):
    to_remove = set()  # store indices of lines to remove

    for i in range(len(lines) - 1):
        line1, line2 = lines[i].strip(), lines[i+1].strip()
        
        ratio = difflib.SequenceMatcher(None, line1, line2).ratio()
        
        if ratio >= threshold:
            # mark the later line (i+1) for removal
            to_remove.add(i+1)
    
    # keep only lines whose index is NOT in to_remove
    cleaned_lines = [line for idx, line in enumerate(lines) if idx not in to_remove]

    return cleaned_lines



def extract_text(pages:list[fitz.Page], start_ID : str, end_ID : str, reading_area: fitz.Rect):
    extracted_text : str = ''
    for page in pages:
        text = page.get_textbox(reading_area)
        start_match = re.search(rf'{re.escape(start_ID)}\n', text, flags=re.IGNORECASE)
        end_match = re.search(rf'{re.escape(end_ID)}\n', text, flags=re.IGNORECASE)

        if start_match != None and end_match != None:
            extracted_text += text[start_match.end() : end_match.start()] + '\n'

        elif start_match != None and end_match == None: 
            extracted_text += text[start_match.end() : ] + '\n'

        elif start_match == None and end_match == None:
            extracted_text += text[ : ] + '\n'

        elif start_match == None and end_match != None:
            extracted_text += text[ : end_match.start()] + '\n'

    return extracted_text


def extract_pages(start_page : int, end_page : int, doc: fitz.Document):
    if end_page == None:
        pages: list = [doc[start_page]]
    else: 
        pages : list = doc[start_page : end_page]

    return pages


def process_toc(doc: fitz.Document):
    """ Analyse  Hierarchy """
    toc = doc.get_toc()

    toc_hierarchy = []
    current_chapter = None
    current_title = None

    for level, name, page in toc:
        if level == 2:
            # New chapter
            current_chapter = {
                "chapter": name,
                "page": page,
                "titles": []
            }
            toc_hierarchy.append(current_chapter)

        elif level == 3:
            # New title under current chapter
            if current_chapter is None:
                # Handle malformed TOC (title before chapter)
                continue
            current_title = {
                "title": name,
                "page": page,
                "sections": []
            }
            current_chapter["titles"].append(current_title)

        elif level == 4:
            # New section under current title
            if current_title is None:
                # Handle malformed TOC (section before title)
                continue
            current_title["sections"].append({
                "section": name,
                "page": page
            })

    return toc_hierarchy
