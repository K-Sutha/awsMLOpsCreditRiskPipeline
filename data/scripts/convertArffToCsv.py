#!/usr/bin/env python3
"""
ARFF to CSV Converter
Converts OpenML ARFF format to CSV for easier processing
"""

import re
import csv
import sys

def parseArffToCsv(inputFile, outputFile):
    """
    Convert ARFF file to CSV format
    """
    print(f"Converting {inputFile} to {outputFile}...")
    
    # Read the ARFF file
    with open(inputFile, 'r') as file:
        content = file.read()
    
    # Find the @DATA section
    dataSection = re.search(r'@DATA\s*\n(.*)', content, re.DOTALL)
    if not dataSection:
        raise ValueError("No @DATA section found in ARFF file")
    
    # Extract data lines
    dataLines = dataSection.group(1).strip().split('\n')
    
    # Find attribute names
    attributePattern = r'@ATTRIBUTE\s+(\w+)\s+\w+'
    attributes = re.findall(attributePattern, content)
    
    if not attributes:
        raise ValueError("No attributes found in ARFF file")
    
    print(f"Found {len(attributes)} attributes: {', '.join(attributes)}")
    print(f"Found {len(dataLines)} data records")
    
    # Write CSV file
    with open(outputFile, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        
        # Write header
        writer.writerow(attributes)
        
        # Write data rows
        for line in dataLines:
            line = line.strip()
            if line and not line.startswith('%'):  # Skip empty lines and comments
                # Split by comma and clean values
                values = [val.strip() for val in line.split(',')]
                writer.writerow(values)
    
    print(f"Successfully converted to {outputFile}")
    print(f"Records processed: {len(dataLines)}")

def main():
    # File paths
    inputFile = "dataset"  # ARFF file
    outputFile = "creditRiskDataset.csv"  # Output CSV
    
    try:
        parseArffToCsv(inputFile, outputFile)
        
        # Display basic info about the converted file
        with open(outputFile, 'r') as file:
            lines = file.readlines()
            print(f"\nCSV File Summary:")
            print(f"Header: {lines[0].strip()}")
            print(f"Total lines: {len(lines)}")
            print(f"Data records: {len(lines) - 1}")
            
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
