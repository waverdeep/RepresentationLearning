from sklearn.model_selection import train_test_split


def get_split_data(data_path, train_output_path, test_output_path):
    data = open(data_path, 'r')
    file_list = [x.strip() for x in data.readlines()]
    x_train, x_test = train_test_split(file_list, test_size=0.2, random_state=777)
    train_output_file = open('{}'.format(train_output_path), 'w')
    for index, file in enumerate(x_train):
        train_output_file.write("{}\n".format(file))
    train_output_file.close()
    test_output_file = open('{}'.format(test_output_path), 'w')
    for index, file in enumerate(x_test):
        test_output_file.write("{}\n".format(file))
    test_output_file.close()


def get_other_split(split_path, output_path):
    data = open(split_path, 'r')
    output = open(output_path, 'w')
    file_list = [x.strip() for x in data.readlines()]

    for index, file in enumerate(file_list):
        temp = file.split('-')
        line = '../../dataset/LibriSpeech/train-clean-100/{}/{}/{}.flac'.format(temp[0], temp[1], file)
        output.write('{}\n'.format(line))

    data.close()
    output.close()


if __name__ == '__main__':
    # print("start converting ... ")
    # get_other_split('../../dataset/train_split.txt', '../../dataset/train-clean-original.txt')
    # get_other_split('../../dataset/test_split.txt', '../../dataset/test-clean-original.txt')
    # print("finish ...")

    # get_split_data('../../dataset/urbansound-20480.txt',
    #                '../../dataset/train-urbansound-20480.txt',
    #                '../../dataset/test-urbansound-20480.txt')
    get_split_data('../../dataset/librispeech360-20480.txt',
                   '../../dataset/train-librispeech360-20480.txt',
                   '../../dataset/test-librispeech360-20480.txt')