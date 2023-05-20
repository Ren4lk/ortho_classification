import os
import cv2


def createMirroredImages(data_dir):
    with os.scandir(data_dir) as dirs:
        dir_paths = [dir.path for dir in dirs]
        for dir_path in dir_paths:
            if dir_path.find('left') == -1 and dir_path.find('right') == -1:
                with os.scandir(dir_path) as photos:
                    for photo in photos:
                        img = cv2.imread(photo.path)
                        img = cv2.flip(img, 1)
                        cv2.imwrite(photo.path[:photo.path.find('.')] +
                                    '_mirrored'+photo.path[photo.path.find('.'):], img)

            elif dir_path.find('left') != -1:
                left_path = dir_path
                right_path = dir_path.replace('left', 'right')
                with os.scandir(left_path) as left, os.scandir(right_path) as right:
                    left_photo_paths = [photo.path for photo in left]
                    right_photo_paths = [photo.path for photo in right]

                for photo in left_photo_paths:
                    img = cv2.imread(photo)
                    img = cv2.flip(img, 1)
                    cv2.imwrite(os.path.join(right_path,
                                             photo[photo.rfind('/')+1:photo.find('.')] +
                                             '_mirrored' + photo[photo.find('.'):]), img)

                for photo in right_photo_paths:
                    img = cv2.imread(photo)
                    img = cv2.flip(img, 1)
                    cv2.imwrite(os.path.join(left_path,
                                             photo[photo.rfind('/')+1:photo.find('.')] +
                                             '_mirrored' + photo[photo.find('.'):]), img)


if __name__ == '__main__':
    createMirroredImages(
        'ortho_classification/sorted_photo_v2_corrected_with_mirrored_images')
