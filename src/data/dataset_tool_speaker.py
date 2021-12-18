from collections import defaultdict


def get_librispeech_speaker_list(file_list):
    speaker_list = []
    for index, file in enumerate(file_list):
        filename = file.split("/")[-1]
        filename = filename.split(".")[0]
        speaker_id = filename.split("-")[0] # speaker_id, dir_id, sample_id
        speaker_list.append(speaker_id)
    return speaker_list


def get_competition_speaker_list(file_list):
    speaker_list = []
    for index, file in enumerate(file_list):
        temp = file.split('/')
        speaker_id = temp[-2]
        speaker_list.append(speaker_id)
    return speaker_list


def get_speaker_dict(speaker_list):
    speaker_id_dict = {}
    for idx, key in enumerate(sorted(list(set(speaker_list)))):
        speaker_id_dict[key] = idx
    return speaker_id_dict


def get_speaker_align(file_list):
    item_list = []
    speaker_align = defaultdict(list)
    index = 0
    for index, file in enumerate(file_list):
        filename = file.split("/")[-1]
        filename = filename.split(".")[0]
        speaker_id, dir_id, sample_id = filename.split("-")
        item_list.append((speaker_id, dir_id, sample_id))
        speaker_align[speaker_id].append(index)
        index += 1

    return item_list, speaker_align
