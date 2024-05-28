#!/bin/bash

######################################################################################
# Script Name: getDexFiles.sh
# Description: Finds all APKs in a directory and extracts their .dex files.
#       Usage: sh detectors/R2-D2/preprocess/getDexFiles.sh \
#                 "/mnt/nas/squeyrut/andromak-datasets/GM19/gw" \
#                 "data/features/R2-D2/original/GM19/gw"
######################################################################################


# Start timer
start=$(date +%s)   

if [ "$1" = "--from-subfolders" ]; then

    # Set the source directories
    source_dir="$2"

    # Set the .dex destination directories
    dex_dir="$3"

    apk_dir="$source_dir"/*/*.apk
else

    # Set the source directories
    source_dir="$1"

    # Set the .dex destination directories
    dex_dir="$2"

    apk_dir="$source_dir"/*.apk
fi

for apk in $apk_dir; do

    base=$(basename "$apk" .apk)

    # Check if the .dex file already exists
    if [ -f "$dex_dir/${base}_classes.dex" ]; then
        echo ".dex file for $base already exists. Skipping..."
        continue
    fi

    # Create a temporary directory to extract the .dex file
    temp_dir=$(mktemp -d)

    # Unzip the APK and extract classes.dex file(s)
    unzip "$apk" "classes*.dex" -d "$temp_dir"

    # Move the extracted .dex files to the destination directory
    for dex in "$temp_dir"/classes*.dex; do
        dex_base=$(basename "$dex")
        mv "$dex" "$dex_dir/${base}_${dex_base}"
    done

    # Remove the temporary directory
    rm -rf "$temp_dir"

done

# End timer
end=$(date +%s)

# Calculate time taken
runtime=$((end-start))
echo "Time taken: $runtime seconds"
