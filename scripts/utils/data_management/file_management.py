import re
import numpy as np
import os
from paramiko import SSHClient
from scp import SCPClient
import yaml
import shutil


def find_pattern_names(string_name, str_pattern):
    """
    Looks for a pattern name in a string and returns the number after it
    :param string_name: the string where to look for a pattern
    :param str_pattern: the pattern that needs to be found
    :return:
    """
    match = re.search(str_pattern + '(\d+)', string_name)
    if match:
        return match.group(1)
    else:
        return np.nan


def check_file_exists(path_list, name_file='list_experiments.txt'):
    list_files = os.listdir(path_list)
    if name_file in list_files:
        list_files = read_txt_file(os.path.join(path_list, name_file))
        return list_files
    else:
        return None


def read_txt_file(dir_file):
    with open(dir_file, 'r') as f:
        lines = f.readlines()
    lines = [f.strip() for f in lines]
    f.close()
    return lines


def compress_files(dir_name, destination_dir=os.path.join(os.getcwd())):
    print(f'Compressing: {dir_name} ... ')
    output_name = os.path.split(os.path.normpath(dir_name))[-1]
    shutil.make_archive(output_name, 'zip', dir_name)
    move_files(output_name + '.zip', destination_dir)


def transfer_files(local_host):

    ssh = SSHClient()
    ssh.load_system_host_keys()
    print(f'connecting to {local_host}')
    ssh.connect(local_host)

    # SCPCLient takes a paramiko transport as an argument
    scp = SCPClient(ssh.get_transport())

    #scp.put('test.txt', 'test2.txt')
    #scp.get('test2.txt')
    scp.close()


def read_yaml_file(path_file):
    with open(path_file) as file:
        document = yaml.full_load(file)

    return document


def move_files(name_file, destination_dir):

    print(f'Moving file {name_file}')
    current_path = os.path.join(os.getcwd(), name_file)
    destination_path = os.path.join(destination_dir, name_file)
    shutil.move(current_path, destination_path)
    print(f'file_saved_at {destination_dir} ')
