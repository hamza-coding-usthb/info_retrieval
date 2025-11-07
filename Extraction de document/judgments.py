# Open the input file 'med_rel.txt' and the output file 'judgments.txt'
with open('med/MED.REL', 'r') as infile, open('Judgements', 'w') as outfile:
    # Loop through each line in the input file
    for line in infile:
        # Split the line into columns based on spaces
        columns = line.split()
        # Write the first and third columns to the output file, separated by a tab
        outfile.write(f"{columns[0]}\t{columns[2]}\n")
