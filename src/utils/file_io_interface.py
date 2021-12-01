import glob
import os
import json


def get_pure_filename(filename):
    temp = filename.split('.')
    del temp[-1]
    temp = '.'.join(temp)
    temp = temp.split('/')
    temp = temp[-1]
    return temp


def get_all_file_path(input_dir, file_extension):
    temp = glob.glob(os.path.join(input_dir, '**', '*.{}'.format(file_extension)), recursive=True)
    return temp


def load_json_config(filename):
    with open(filename, 'r') as configuration:
        config = json.load(configuration)
    return config


def make_directory(directory_name, format_logger=None):
    try:
        if not os.path.exists(directory_name):
            os.makedirs(directory_name)
    except OSError:
        if format_logger is not None:
            format_logger.info('Error: make directory: {}'.format(directory_name))
        else:
            print('Error: make directory: {}'.format(directory_name))