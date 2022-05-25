import os
from absl import app, flags
from absl.flags import FLAGS

def main(_argv):
    pass

if __name__ == '__main__':

    flags.DEFINE_string('base_path', None, 'base directory')
    flags.DEFINE_string('output_file_name', 'list_experiments.txt', 'name of the output file')
    try:
        app.run(main)
    except SystemExit:
        pass