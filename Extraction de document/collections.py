import os

# Path to the input file and the output folder
input_file = 'med/MED.ALL'
output_folder = 'Collection'

# Make sure the output folder exists
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Read the content of the input file
with open(input_file, 'r') as file:
    content = file.read()

# Split the content based on ".I <id>" to separate each document
documents = content.split('.I ')[1:]  # Skip the first part before the first document

for doc in documents:
    # Split the document by the first occurrence of ".W" to separate title/content
    doc_parts = doc.split('.W', 1)
    doc_id = doc_parts[0].strip()  # Extract document ID
    doc_text = doc_parts[1].strip() if len(doc_parts) > 1 else ""  # Extract document text after .W

    # Construct the file name (D<ID>.txt)
    file_name = f"D{doc_id}.txt"
    file_path = os.path.join(output_folder, file_name)

    # Write the document text into the corresponding file
    with open(file_path, 'w') as doc_file:
        doc_file.write(doc_text)

    print(f"Created file: {file_path}")
