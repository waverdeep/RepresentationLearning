from sklearn.model_selection import train_test_split
import interface_file_io as file_io


def make_train_test_file(data_path, train_file_path, test_file_path):
    file_list = file_io.read_txt2list(data_path)
    x_train, x_test = train_test_split(file_list, test_size=0.2, random_state=777)
    file_io.make_list2txt(x_train, train_file_path)
    file_io.make_list2txt(x_test, test_file_path)


def make_cpc_paper_train_test_file(split_path, output_path):
    data = open(split_path, 'r')
    output = open(output_path, 'w')
    file_list = [x.strip() for x in data.readlines()]

    for index, file in enumerate(file_list):
        temp = file.split('-')
        line = '../../dataset/LibriSpeech/train-clean-100/{}/{}/{}.flac'.format(temp[0], temp[1], file)
        output.write('{}\n'.format(line))

    data.close()
    output.close()


def make_audio_filelist():
    pass


if __name__ == '__main__':
    task = "train_test_split"
    if task == "train_test_split":
        # make_train_test_file('../../dataset/urbansound-20480.txt',
        #                      '../../dataset/train-urbansound-20480.txt',
        #                      '../../dataset/test-urbansound-20480.txt')
        # make_train_test_file('../../dataset/librispeech360-20480.txt',
        #                      '../../dataset/train-librispeech360-20480.txt',
        #                      '../../dataset/test-librispeech360-20480.txt')
        make_train_test_file('../../dataset/kspon-20480.txt',
                             '../../dataset/train-kspon-20480.txt',
                             '../../dataset/test-kspon-20480.txt')

    elif task == "make_cpc_paper_train_test_file":
        # print("start converting ... ")
        # get_other_split('../../dataset/train_split.txt', '../../dataset/train-clean-original.txt')
        # get_other_split('../../dataset/test_split.txt', '../../dataset/test-clean-original.txt')
        # print("finish ...")
        pass