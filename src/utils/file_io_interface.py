import glob
import os


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
