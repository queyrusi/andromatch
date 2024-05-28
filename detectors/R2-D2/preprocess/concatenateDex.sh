#!/bin/bash

######################################################################################
# Script Name: concatenateDex.sh
# Description: This script is used to concatenate the .dex files extracted from APKs 
#              in a directory as an APK sometimes yield several .dex files.
#              It takes two arguments: the input directory containing the .dex files 
#              and the output directory where the concatenated .dex file will be saved.
# Usage: sudo ./detectors/R2-D2/preprocess/concatenate_dex.sh <input_directory> \
#        <output_directory>
# Example: sudo ./detectors/R2-D2/preprocess/concatenateDex.sh \
#          "data/features/original/GM19/gw/R2-D2/dex/" \
#          "data/features/original/GM19/gw/R2-D2/dex_concatenate/"
######################################################################################


# Check for the correct number of arguments
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <input_folder> <output_folder>"
    exit 1
fi

input_folder="$1"
output_folder="$2"

# Create the output directory if it doesn't exist
mkdir -p "$output_folder"

# Create a temporary directory for concatenation
temp_dir=$(mktemp -d)

# Function to extract the hash part of the filename
extract_hash() {
    local filename="$1"
    echo "$(basename "$filename" | cut -d'_' -f1)"
}

# Find all .dex files in the input folder
find "$input_folder" -type f -name '*.dex' | while read -r dex_file; do
    hash=$(extract_hash "$dex_file")
    dex_files=($(find "$input_folder" -type f -name "${hash}_*.dex"))

    if [ "${#dex_files[@]}" -gt 1 ]; then
        # Concatenate all dex files with the same hash
        concatenated_file="$output_folder/${hash}_classes_concatenated.dex"
        cat "${dex_files[@]}" > "$temp_dir/${hash}_classes_concatenated.dex"
        mv "$temp_dir/${hash}_classes_concatenated.dex" "$concatenated_file"
    else
        # Move the unique dex file to the output folder
        mv "$dex_file" "$output_folder/"
    fi
done

# Clean up the temporary directory
rm -rf "$temp_dir"

echo "Processing completed. All files have been moved to $output_folder."