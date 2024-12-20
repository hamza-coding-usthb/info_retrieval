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

def scalar_product(query_terms, query_weights, inverse_data):
    scores = defaultdict(float)
    for term in query_terms:
        if term in query_weights and term in inverse_data:
            query_weight = query_weights[term]
            for doc_id, (_, doc_weight) in inverse_data[term].items():
                scores[doc_id] += query_weight * doc_weight
    return scores

"""
# Compute RSV using Cosine Similarity
def cosine_similarity(query_terms, query_weights, inverse_data):
    scores = defaultdict(float)
    query_magnitude = 0.0
    doc_magnitudes = defaultdict(float)

    for term in query_terms:
        if term in query_weights and term in inverse_data:
            query_weight = query_weights[term]
            query_magnitude += query_weight ** 2
            for doc_id, (_, doc_weight) in inverse_data[term].items():
                scores[doc_id] += query_weight * doc_weight  # Dot product
                doc_magnitudes[doc_id] += doc_weight ** 2  # Document magnitude

    query_magnitude = math.sqrt(query_magnitude)  # Final query magnitude

    for doc_id in scores:
        if doc_magnitudes[doc_id] > 0:
            scores[doc_id] /= (query_magnitude * math.sqrt(doc_magnitudes[doc_id]))

    return scores

"""



def cosine_similarity(query_terms, query_weights, inverse_data):
  
    scores = defaultdict(float)
    doc_magnitudes = defaultdict(float)
    query_magnitude = 0.0

    # Compute dot product and accumulate document magnitudes
    for term in query_terms:
        if term in query_weights and term in inverse_data:
            query_weight = query_weights[term]
            query_magnitude += query_weight ** 2  # Accumulate query magnitude

            for doc_id, (_, doc_weight) in inverse_data[term].items():
                # Dot product component
                scores[doc_id] += query_weight * doc_weight
                # Document magnitude component
                doc_magnitudes[doc_id] += doc_weight ** 2

    # Compute final query magnitude
    query_magnitude = math.sqrt(query_magnitude)

    # Normalize scores with query and document magnitudes
    for doc_id in scores:
        if doc_magnitudes[doc_id] > 0 and query_magnitude > 0:
            scores[doc_id] /= (query_magnitude * math.sqrt(doc_magnitudes[doc_id]))
        else:
            scores[doc_id] = 0.0  # Ensure no division by zero

    return scores
"""


# Compute RSV using Jaccard Index
def jaccard_index(query_terms, descriptor_data):
    scores = defaultdict(float)
    for doc_id, terms in descriptor_data.items():
        doc_terms = set(terms.keys())
        intersection = len(query_terms & doc_terms)
        union = len(query_terms | doc_terms)
        if union > 0:
            scores[doc_id] = intersection / union
    return {doc_id: score for doc_id, score in scores.items() if score > 0.0}
"""


def weighted_jaccard_index(query_weights, descriptor_data):
    """
    Compute RSV using the weighted Jaccard variation.

    Parameters:
    - query_weights: Dictionary {term: weight} for the query.
    - descriptor_data: Dictionary {doc_id: {term: (freq, weight)}} for all documents.

    Returns:
    - scores: Dictionary {doc_id: weighted_jaccard_score} for documents with scores > 0.
    """
    scores = defaultdict(float)

    for doc_id, doc_terms in descriptor_data.items():
        dot_product = 0.0
        query_magnitude_squared = 0.0
        doc_magnitude_squared = 0.0

        # Compute dot product and magnitudes
        for term, query_weight in query_weights.items():
            query_magnitude_squared += query_weight ** 2  # Sum of squared query weights
            if term in doc_terms:
                # Extract the weight from (freq, weight)
                _, doc_weight = doc_terms[term]
                dot_product += query_weight * doc_weight  # Weighted intersection

        for _, (_, doc_weight) in doc_terms.items():
            doc_magnitude_squared += doc_weight ** 2  # Sum of squared document weights

        # Compute weighted union
        weighted_union = query_magnitude_squared + doc_magnitude_squared - dot_product

        # Avoid division by zero
        if weighted_union > 0:
            scores[doc_id] = dot_product / weighted_union

    # Filter out documents with a score of 0
    return {doc_id: score for doc_id, score in scores.items() if score > 0.0}

# Main function to compute RSV
def compute_rsv(query, descriptor_file, inverse_file):
    """
    Compute RSV scores for a given query using Scalar Product, Cosine Similarity, and Jaccard Index.

    Parameters:
    - query: List of query terms.
    - descriptor_file: Path to the descriptor file.
    - inverse_file: Path to the inverse index file.

    Returns:
    - scalar_scores, cosine_scores, jaccard_scores: Dictionaries of scores for each method.
    """
    try:
        # Load descriptor and inverse index data
        descriptor_data = load_data(descriptor_file, use_index=False)
        inverse_data = load_data(inverse_file, use_index=True)
    except FileNotFoundError as e:
        raise FileNotFoundError(f"Error loading files: {e}")
    except Exception as e:
        raise ValueError(f"Unexpected error loading data: {e}")

    # Convert query to a set for set operations
    query_terms = set(query)
    query_weights = compute_query_weights(query)

    # Compute scores for each RSV method
    scalar_scores = scalar_product(query_terms, query_weights, inverse_data)
    cosine_scores = cosine_similarity(query_terms, query_weights, inverse_data)
    jaccard_scores = weighted_jaccard_index(query_weights, descriptor_data)  


    return scalar_scores, cosine_scores, jaccard_scores

def compute_query_weights(query_terms):
    term_freq = defaultdict(int)
    for term in query_terms:
        term_freq[term] += 1
    query_weights = {term: freq / len(query_terms) for term, freq in term_freq.items()}
    return query_weights
