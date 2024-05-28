#!/bin/bash

#!/bin/bash

######################################################################################
# Script Name: dexFolder2Images.sh
# Description: This script converts the DEX files in a folder to images. It is part of
#              the preprocessing step in the R2-D2 detector of the AndroMatch project.
#              The script takes a folder containing DEX files as input and converts 
#              each DEX file to an image representation using a specific conversion 
#              algorithm. The resulting images can then be used as input for further
#              analysis and processing in the R2-D2 detector.
# Usage: ./detectors/R2-D2/preprocess/dexFolder2Images.sh <input_directory> \
#        <output_directory>
# Example: nohup ./detectors/R2-D2/preprocess/dexFolder2Images.sh \
#          data/features/original/GM19/mw/R2-D2/dex_concatenate/ \
#          data/features/original/GM19/mw/R2-D2/images &
######################################################################################


# Check if the correct number of arguments are provided
if [ "$#" -ne 2 ]; then
    echo "Usage: ./script.sh input_folder output_folder"
    exit 1
fi

input_folder=$1
output_folder=$2

# Find all dex files in the input folder and run the python script on them
find "$input_folder" -name "*.dex" -type f | while read -r file
do
    python3 detectors/R2-D2/preprocess/dex2Image.py "$file" "$output_folder"
done