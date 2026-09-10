import os
import re

def search_files(directory, pattern):
    compiled_pattern = re.compile(pattern, re.IGNORECASE)
    for root, dirs, files in os.walk(directory):
        # Skip directories like .git or __pycache__
        if any(ignored in root for ignored in ['.git', '__pycache__', '.ipynb_checkpoints', '.agents']):
            continue
        for file in files:
            if file.endswith(('.py', '.ipynb', '.txt', '.csv')):
                filepath = os.path.join(root, file)
                try:
                    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                        for line_num, line in enumerate(f, 1):
                            if compiled_pattern.search(line):
                                print(f"{file}:{line_num}: {line.strip()[:100]}")
                except Exception as e:
                    pass

if __name__ == '__main__':
    search_directory = r"c:\Users\josep\Documents\MRes mini-project 2"
    print("Searching for 'wavelength'...")
    search_files(search_directory, r'wavelength')
    print("\nSearching for 'frequency'...")
    search_files(search_directory, r'frequency')
