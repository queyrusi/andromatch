#!/bin/bash

apk_folder="$1"
feature_output_folder="$2"

# Create the output folder if it doesn't exist
mkdir -p "$feature_output_folder"

# Create the dex and dex_concatenate folders beside the output folder
mkdir -p "${feature_output_folder}/../dex"
mkdir -p "${feature_output_folder}/../dex_concatenate"

echo "Directories created:"
echo "$feature_output_folder"
echo "${feature_output_folder}/../dex"
echo "${feature_output_folder}/../dex_concatenate"

# Start timer
start=$(date +%s)

sh getDexFiles.sh "$apk_folder" "${feature_output_folder}/../dex"
./concatenateDex.sh "${feature_output_folder}/../dex" "${feature_output_folder}/../dex_concatenate"
./dexFolder2Images.sh "${feature_output_folder}/../dex_concatenate" "${feature_output_folder}"

# End timer
end=$(date +%s)

# Calculate time taken
runtime=$((end-start))

echo "Time taken: $runtime seconds"

# Delete the dex and dex_concatenate folders
rm -rf "${feature_output_folder}/../dex"
rm -rf "${feature_output_folder}/../dex_concatenate"

echo "Directories deleted:"
echo "${feature_output_folder}/../dex"
echo "${feature_output_folder}/../dex_concatenate"

# Count the number of images created
num_images=$(ls -1q "${feature_output_folder}" | wc -l)
echo "Number of images created: $num_images"
