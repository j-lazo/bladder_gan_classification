import cv2
from matplotlib import pyplot as plt
from absl import app, flags
from absl.flags import FLAGS
import os


def main(_argv):

    path_video_file = FLAGS.path_video_file
    save_dir = FLAGS.save_dir

    if not(os.path.isfile(path_video_file)):
        raise f'Video path:{path_video_file} not found'

    dir_video = os.path.split(path_video_file)[0]
    name_video = os.path.split(path_video_file)[-1]
    file_extension = name_video.split('.')[-1]
    case_id = name_video.replace('.' + file_extension, '')

    if save_dir == '':
        save_dir = os.path.join(dir_video, case_id)
    else:
        save_dir = save_dir + case_id

    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)

    print(path_video_file)
    vidcap = cv2.VideoCapture(path_video_file)
    success, image = vidcap.read()
    count = 0

    while success:
        save_name = ''.join([save_dir, '/', name_video, '_', '{:0>4}'.format(count), ".jpg"])
        """print(count, save_name)
        plt.figure()
        plt.imshow(image)
        plt.show()"""
        cv2.imwrite(save_name, image)  # save frame as JPEG file
        success, image = vidcap.read()
        print('Reading a new frame: ', success)
        count += 1


if __name__ == '__main__':
    flags.DEFINE_string('path_video_file', '', 'path of the video')
    flags.DEFINE_string('save_dir', '', 'path where to save the images extracted')

    try:
        app.run(main)
    except SystemExit:
        pass