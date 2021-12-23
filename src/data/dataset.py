import torchaudio
from torch.utils import data
import src.data.dataset_librispeech as librispeech
import src.data.dataset_voxceleb as voxceleb
import src.data.dataset_normal as normal
torchaudio.set_audio_backend("sox_io")


def get_dataloader(config, mode='train'):
    dataset = None
    dataset_type = config['dataset_type']
    if dataset_type == 'LibriSpeechWaveformDataset':
        dataset = librispeech.LibriSpeechWaveformDataset(
            directory_path=config['{}_dataset'.format(mode)],
            audio_window=config['audio_window'],
        )
    elif dataset_type == 'LibriSpeechFullWaveformDataset':
        dataset = librispeech.LibriSpeechWaveformDataset(
            directory_path=config['{}_dataset'.format(mode)],
            audio_window=config['audio_window'],
            full_audio=True,
        )
    elif dataset_type == 'VoxWaveformDataset':
        dataset = voxceleb.VoxWaveformDataset(
            directory_path=config['{}_dataset'.format(mode)],
            audio_window=config['audio_window'],
        )
    else:
        dataset = normal.NormalWaveformDataset(
            directory_path=config['{}_dataset'.format(mode)],
            audio_window=config['audio_window'],
        )

    dataloader = data.DataLoader(
        dataset=dataset,
        batch_size=config['batch_size'],
        shuffle=config['dataset_shuffle'],
        num_workers=config['num_workers'],
        pin_memory=config['pin_memory'],
    )

    return dataloader, dataset


if __name__ == '__main__':
    # config = {
    #     "audio_window": 20480,
    #     "batch_size": 1,
    #     # dataset
    #     "dataset_type": "WaveformDataset",
    #     "train_dataset": "../../dataset/baseline-train-split.txt",
    #     "test_dataset": "../../dataset/baseline-test-split.txt",
    #     "num_workers": 16,
    #     "dataset_shuffle": True,
    #     "pin_memory": True,
    # }
    # train_loader, train_dataset = get_dataloader(config=config, mode='train')
    # for data in train_loader:
    #     _, out_filename, speaker_id = data
    #     print(out_filename)
    #     print(speaker_id)
    #     break
    # speaker_id_dict = {}
    # print(len(list(set(train_dataset.speaker_list))))
    # for idx, key in enumerate(sorted(list(set(train_dataset.speaker_list)))):
    #     speaker_id_dict[key] = idx
    # print(speaker_id_dict)

    # get_dataloader_speaker_classification(
    #     directory_path='../../dataset/test-list-librispeech.txt',
    #     audio_window=20480,
    #     batch_size=8,
    #     num_workers=8,
    #     shuffle=True,
    #     pin_memory=False,
    #     speaker_index_file='../../dataset/test-speaker-list-librispeech.txt'
    # )

    config = {
        "audio_window": 20480,
        "batch_size": 1,
        # dataset
        "dataset_type": "LibriSpeechFullWaveformDataset",
        "train_dataset": "../../dataset/baseline-train-split.txt",
        "test_dataset": "../../dataset/baseline-test-split.txt",
        "num_workers": 16,
        "dataset_shuffle": True,
        "pin_memory": True,
    }
    train_loader, train_dataset = get_dataloader(config=config, mode='train')
    for data in train_loader:
        _waveform, _out_filename, _speaker_id = data

        break


