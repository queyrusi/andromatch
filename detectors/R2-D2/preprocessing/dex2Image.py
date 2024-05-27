import sys
import os
import math
from PIL import Image


def extract_filename(path):
    # Filename looks like [HASH]_classes.dex
    return os.path.basename(path).split("_")[0]

def generate_png(stream: bytes, filename: str, folder: str):
    current_len = len(stream)
    image = Image.frombytes(mode='L', size=(1, current_len), data=stream)
    trimmed_filename = extract_filename(filename)
    image.save(os.path.join(folder, f'{trimmed_filename}.png'))
    

if __name__ == "__main__":
    if len(sys.argv) < 3:
        raise Exception("[!] Usage: python3 dex2Image.py DEX DESTINATION")
    else:
        filename = sys.argv[1]
        destination_folder = sys.argv[2]
    try:
        with open(filename, 'rb') as file:
            stream = file.read()
        generate_png(stream, filename, destination_folder)
        print(f"Image successfully generated from {filename}")
    except Exception as e:
        print("[!] An exception occured with: {}".format(filename))
        print("Exception: {}".format(e))
