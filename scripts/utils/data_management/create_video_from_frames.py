import cv2
import numpy as np
from absl import app, flags
from absl.flags import FLAGS
import os
import tqdm


def main(_argv):
    path_frames_1 = FLAGS.path_frames_1
    path_frames_2 = FLAGS.path_frames_2

    list_imgs_1 = sorted(os.listdir(path_frames_1))
    output_dir = os.path.split(os.path.normpath(path_frames_1))[0]

    if path_frames_2:
        list_imgs_2 = sorted(os.listdir(path_frames_2))
        if len(list_imgs_1) != len(list_imgs_2):
            raise ValueError(f'mistmatch of number of images {len(list_imgs_1)} in path 1, '
                             f'{len(list_imgs_2)} in path 2')

    img_array = []
    for j, filename in enumerate(tqdm.tqdm(list_imgs_1[:], desc='Reading images')):
        img1 = cv2.imread(os.path.join(path_frames_1, filename))
        aa_milne_arr = ['Low Grade', 'High Grade', 'NTL']
        text = np.random.choice(aa_milne_arr, 1, p=[0.8, 0.15, 0.05])
        img1 = cv2.putText(img1, text[0], (500, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        if path_frames_2:
            img2 = cv2.imread(os.path.join(path_frames_2, list_imgs_2[j]))
            img_array.append(np.vstack((img1, img2)))
        else:
            img_array.append(img1)

    path_video_out = os.path.join(output_dir, 'video.mp4')
    fourcc = cv2.VideoWriter_fourcc(*'MPEG')
    out = cv2.VideoWriter(path_video_out, fourcc, 15, (img_array[0].shape[1], img_array[0].shape[0]))

    for i, image in enumerate(tqdm.tqdm(img_array[:], desc='Creating Video')):
        out.write(img_array[i])
    out.release()


if __name__ == '__main__':
    flags.DEFINE_string('path_frames_1', '', 'path of the frames to make the video')
    flags.DEFINE_string('path_frames_2', None, 'path of the frames to make the video')

    try:
        app.run(main)
    except SystemExit:
        pass