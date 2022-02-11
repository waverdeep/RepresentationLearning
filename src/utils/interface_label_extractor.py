import interface_file_io as file_io

if __name__ == '__main__':
    filelist = "../../dataset/voxceleb01-20480.txt"
    dataset = file_io.read_txt2list(filelist)
    label = []
    for data in dataset:
        label.append(data.split('/')[5])
    label = list(set(label))
    file_io.make_list2txt(label, "../../dataset/voxceleb01-label.txt")