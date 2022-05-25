import os
from absl import app, flags
from absl.flags import FLAGS


def read_file(dir_file):
    with open(dir_file, 'r') as f:
        lines = f.readlines()
    lines = [f.strip() for f in lines]
    f.close()
    return lines

def main(_argv):
    remote = FLAGS.remote
    local = FLAGS.local
    local_path_results = FLAGS.local_path_results
    name_file_remote = FLAGS.name_file_remote
    name_file_local = FLAGS.name_file_local

    list_files = os.listdir(local_path_results)
    if name_file_local in list_files:
        list_files_local = read_file(os.path.join(local_path_results, name_file_local))
        list_files_remote = read_file(os.path.join(local_path_results, name_file_remote))
        list_missing = [f for f in list_files_remote if f not in list_files_local]
        print(list_missing)


if __name__ == '__main__':

    flags.DEFINE_string('remote', None, 'dir to remote')
    flags.DEFINE_string('local', None, 'dir to local')
    flags.DEFINE_string('local_path_results', os.path.join(os.getcwd(), 'results'), 'dir to local results')
    flags.DEFINE_string('name_file_local', 'list_experiments.txt', 'dir to local results')
    flags.DEFINE_string('name_file_remote', 'list_remote_experiments.txt', 'results'), 'dir to local results')

    try:
        app.run(main)
    except SystemExit:
        pass