import os


data_dir = 'sorted_photo_v2_corrected_with_mirrored_images'
result_file = os.path.join(os.path.dirname(
    os.path.abspath(__file__)), 'annotation.txt')


with os.scandir(data_dir) as dirs, open(result_file, 'a') as out_file:
    for dir in dirs:
        with os.scandir(dir.path) as photos:
            for photo in photos:
                target = dir.name if dir.name.find('portrait') == -1 else 'portrait'
                out_file.write(f'{photo.path}\t{target}\n')
