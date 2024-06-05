import os

input_folder = 'data/plate_vehicular/'
counter = 1

for filename in sorted(os.listdir(input_folder)):
    if filename.lower().endswith('.jpg') or filename.lower().endswith('.png'):
        new_filename = str(counter).zfill(4) + os.path.splitext(filename)[1]
        old_filepath = os.path.join(input_folder, filename)
        new_filepath = os.path.join(input_folder, new_filename)

        os.rename(old_filepath, new_filepath)
        counter += 1