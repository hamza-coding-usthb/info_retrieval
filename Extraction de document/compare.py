# Open the input files 'med_rel.txt' and 'judgments.txt' for comparison
with open('med_rel.txt', 'r') as file1, open('judgments.txt', 'r') as file2:
    file1_lines = file1.readlines()  # Read all lines from 'med_rel.txt'
    file2_lines = file2.readlines()  # Read all lines from 'judgments.txt'

# Check if the number of lines is the same
if len(file1_lines) == len(file2_lines):
    print("The files has the same number of lines.")
