import os
from PIL import Image
import pillow_heif


def convert_heic_to_png(heic_path, png_path):
    try:
        # Open HEIC file
        heif_file = pillow_heif.read_heif(heic_path)

        # Convert HEIC file to image data
        image = Image.frombytes(
            heif_file.mode,
            heif_file.size,
            heif_file.data,
            "raw",
            heif_file.mode,
            heif_file.stride,
        )

        # Save image as PNG
        image.save(png_path, "PNG")
        print(f"Converted {heic_path} to {png_path}")
    except Exception as e:
        print(f"Failed to convert {heic_path}: {e}")


def batch_convert_heic_to_png(heic_dir, png_dir):
    # Ensure the output directory exists
    if not os.path.exists(png_dir):
        os.makedirs(png_dir)

    # Iterate over all HEIC files in the input directory
    for filename in os.listdir(heic_dir):
        if filename.lower().endswith(".heic"):
            heic_path = os.path.join(heic_dir, filename)
            png_filename = os.path.splitext(filename)[0] + ".png"
            png_path = os.path.join(png_dir, png_filename)

            # Convert the HEIC image to PNG
            convert_heic_to_png(heic_path, png_path)


# Example usage
heic_dir = "../data/plate_vehicular"  # Directory containing HEIC files
png_dir = "../data"  # Directory to save PNG files
batch_convert_heic_to_png(heic_dir, png_dir)