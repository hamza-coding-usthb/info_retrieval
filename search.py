import os
from collections import defaultdict

def load_file(filename):
    """Helper function to load and read files from the current directory."""
    if not os.path.exists(filename):
        raise FileNotFoundError(f"File {filename} not found.")
    with open(filename, "r", encoding="utf-8") as f:
        return f.read().strip().splitlines()

def parse_line(line, use_index):
    """Helper function to parse a descriptor or index line."""
    parts = line.split()
    if use_index:
        if len(parts) >= 4:
            term, doc_id, freq, weight = parts
            return term, doc_id, int(freq), float(weight)
    else:
        if len(parts) >= 4:
            doc_id, term, freq, weight = parts
            return doc_id, term, int(freq), float(weight)
    return None

def load_data(filename, use_index=False):
    """Load the descriptor or inverse index into a structured format."""
    data = load_file(filename)
    structured_data = defaultdict(dict)
    for line in data:
        parsed = parse_line(line, use_index)
        if parsed:
            if use_index:
                term, doc_id, freq, weight = parsed
                structured_data[term][doc_id] = (freq, weight)
            else:
                doc_id, term, freq, weight = parsed
                structured_data[doc_id][term] = (freq, weight)
    return structured_data

def load_descriptor_or_index(query, search_type, token_method, stem_method, use_index):
    """Load data based on user selection (descriptors or indices)."""
    file_prefix = "Inverse" if use_index else "Descriptor"
    filename = f"{file_prefix}{token_method}{stem_method}.txt"
    data = load_file(filename)

    results = []
    for line in data:
        parsed = parse_line(line, use_index)
        if parsed:
            if search_type == "terms_per_doc" and parsed[1] == query:
                results.append(parsed)
            elif search_type == "docs_per_term" and parsed[0] == query:
                results.append(parsed)
    return results
