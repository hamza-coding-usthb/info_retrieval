import ttkbootstrap as ttk
from ttkbootstrap.constants import *
from tkinter import messagebox
from search import load_descriptor_or_index
from rsv_methods import compute_rsv

def search():
    query = query_entry.get().strip().lower().split()
    if not query:
        messagebox.showerror("Error", "Please enter a query.")
        return

    search_type = "terms_per_doc" if search_option_var.get() == 1 else "docs_per_term"
    token_method = "Split" if split_rb_var.get() == 1 else "Token"
    stem_method = stem_options[stem_rb_var.get()]
    use_index = index_var.get() == 1

    if match_rsv_var.get():
        # Perform RSV match
        descriptor_file = "DescriptorTokenPorter.txt"
        inverse_file = "InverseTokenPorter.txt"
        results = process_rsv_results(query, descriptor_file, inverse_file)
        update_rsv_table(results)
    else:
        # Perform descriptor/inverse index search
        results = load_descriptor_or_index(query[0], search_type, token_method, stem_method, use_index)
        update_results_table(results, search_type)

def process_rsv_results(query, descriptor_file, inverse_file):
    try:
        scalar_scores, cosine_scores, jaccard_scores = compute_rsv(query, descriptor_file, inverse_file)
        return {
            "Scalar Product": scalar_scores,
            "Cosine Similarity": cosine_scores,
            "Jaccard Index": jaccard_scores
        }
    except Exception as e:
        messagebox.showerror("Error", f"Failed to compute RSV: {e}")
        return {}

# Function to update the results table dynamically for standard search
def update_results_table(data, search_type):
    results_table.delete(*results_table.get_children())
    if search_type == "terms_per_doc":
        results_table.config(columns=("№", "Term", "Doc", "Freq", "Poids"))
        for i, (term, doc_id, freq, weight) in enumerate(data, start=1):
            results_table.insert("", "end", values=(i, doc_id, term, freq, weight))
    else:
        results_table.config(columns=("№", "Doc", "Term", "Freq", "Poids"))
        for i, (doc_id, term, freq, weight) in enumerate(data, start=1):
            results_table.insert("", "end", values=(i, term, doc_id, freq, weight))

# Function to update the results table for RSV results
def update_rsv_table(results):
    results_table.delete(*results_table.get_children())
    for method, scores in results.items():
        results_table.insert("", "end", values=(method, "", "", "", ""))
        for doc_id, score in sorted(scores.items(), key=lambda x: x[1], reverse=True):
            results_table.insert("", "end", values=("", doc_id, "", "", round(score, 4)))

# Main application window
root = ttk.Window(themename="superhero")
root.title("Search Interface with RSV")
root.geometry("900x600")

# Wrap everything in a frame with padding
main_frame = ttk.Frame(root, padding=10)
main_frame.pack(fill=BOTH, expand=True)

# Query Section
query_frame = ttk.Frame(main_frame)
query_frame.pack(fill=X, pady=10)
ttk.Label(query_frame, text="Query", font=("Helvetica", 12)).pack(side=LEFT, padx=5)
query_entry = ttk.Entry(query_frame, width=40)
query_entry.pack(side=LEFT, padx=5)
search_button = ttk.Button(query_frame, text="Search", command=search, style="success.TButton")
search_button.pack(side=LEFT, padx=5)

# Search Options Section
options_frame = ttk.Frame(main_frame)
options_frame.pack(fill=X, pady=10)
search_option_var = ttk.IntVar(value=1)
ttk.Radiobutton(options_frame, text="Terms per Doc", variable=search_option_var, value=1).pack(side=LEFT, padx=10)
ttk.Radiobutton(options_frame, text="Docs per Term", variable=search_option_var, value=2).pack(side=LEFT, padx=10)
index_var = ttk.IntVar(value=0)
ttk.Checkbutton(options_frame, text="Index", variable=index_var).pack(side=LEFT, padx=10)
match_rsv_var = ttk.IntVar(value=0)
ttk.Checkbutton(options_frame, text="Match RSV", variable=match_rsv_var).pack(side=LEFT, padx=10)

# Processing Section
processing_frame = ttk.Labelframe(main_frame, text="Processing", padding=10)
processing_frame.pack(fill=X, pady=10)

# Tokenization Frame
tokenization_frame = ttk.Labelframe(processing_frame, text="Tokenization", padding=10)
tokenization_frame.pack(side=LEFT, fill=X, expand=True, padx=10)
split_rb_var = ttk.IntVar(value=1)
ttk.Radiobutton(tokenization_frame, text="Split", variable=split_rb_var, value=1).pack(anchor="w", pady=2)
ttk.Radiobutton(tokenization_frame, text="RegExp", variable=split_rb_var, value=2).pack(anchor="w", pady=2)

# Normalization Frame
normalization_frame = ttk.Labelframe(processing_frame, text="Normalization", padding=10)
normalization_frame.pack(side=LEFT, fill=X, expand=True, padx=10)
stem_rb_var = ttk.IntVar(value=1)
stem_options = {1: "", 2: "Porter", 3: "Lancaster"}
ttk.Radiobutton(normalization_frame, text="No stem", variable=stem_rb_var, value=1).pack(anchor="w", pady=2)
ttk.Radiobutton(normalization_frame, text="Porter Stemmer", variable=stem_rb_var, value=2).pack(anchor="w", pady=2)
ttk.Radiobutton(normalization_frame, text="Lancaster Stemmer", variable=stem_rb_var, value=3).pack(anchor="w", pady=2)

# Results Table Section
results_frame = ttk.Frame(main_frame)
results_frame.pack(fill=BOTH, expand=True, pady=10)
columns = ("Method", "Doc ID", "Term", "Freq", "Poids")
results_table = ttk.Treeview(results_frame, columns=columns, show="headings", height=15, bootstyle="info")
for col in columns:
    results_table.heading(col, text=col)
    results_table.column(col, width=120, anchor="center")
results_table.pack(side=LEFT, fill=BOTH, expand=True)

scrollbar = ttk.Scrollbar(results_frame, orient="vertical", command=results_table.yview)
results_table.configure(yscroll=scrollbar.set)
scrollbar.pack(side=RIGHT, fill=Y)

# Run the application
root.mainloop()