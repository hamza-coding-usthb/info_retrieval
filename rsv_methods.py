import math
from collections import defaultdict

# Load descriptor or inverse index files
def load_file(filename):
    if not filename:
        raise FileNotFoundError(f"File not specified.")
    with open(filename, "r", encoding="utf-8") as f:
        return f.read().strip().splitlines()

# Parse descriptor or inverse index lines
def parse_line(line, use_index=False):
    parts = line.split()
    if use_index:
        term, doc_id, freq, weight = parts
        return term, doc_id, int(freq), float(weight)
    else:
        doc_id, term, freq, weight = parts
        return doc_id, term, int(freq), float(weight)

# Load descriptor or inverse index data
def load_data(filename, use_index=False):
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

# Compute RSV using Scalar Product
def scalar_product(query_terms, inverse_data):
    scores = defaultdict(float)
    for term in query_terms:
        for doc_id, (_, weight) in inverse_data.get(term, {}).items():
            scores[doc_id] += weight
    return scores

# Compute RSV using Cosine Similarity
def cosine_similarity(query_terms, inverse_data):
    scores = defaultdict(float)
    query_magnitude = math.sqrt(len(query_terms))
    doc_magnitudes = defaultdict(float)

    for term in query_terms:
        for doc_id, (_, weight) in inverse_data.get(term, {}).items():
            scores[doc_id] += weight
            doc_magnitudes[doc_id] += weight ** 2

    for doc_id in scores:
        scores[doc_id] /= (query_magnitude * math.sqrt(doc_magnitudes[doc_id]))
    return scores

# Compute RSV using Jaccard Index
def jaccard_index(query_terms, descriptor_data):
    scores = defaultdict(float)
    for doc_id, terms in descriptor_data.items():
        doc_terms = set(terms.keys())
        intersection = len(query_terms & doc_terms)
        union = len(query_terms | doc_terms)
        scores[doc_id] = intersection / union if union > 0 else 0.0
    return scores

# Main function to compute RSV
def compute_rsv(query, descriptor_file, inverse_file):
    query_terms = set(query)
    descriptor_data = load_data(descriptor_file, use_index=False)
    inverse_data = load_data(inverse_file, use_index=True)

    scalar_scores = scalar_product(query_terms, inverse_data)
    cosine_scores = cosine_similarity(query_terms, inverse_data)
    jaccard_scores = jaccard_index(query_terms, descriptor_data)

    return scalar_scores, cosine_scores, jaccard_scores
