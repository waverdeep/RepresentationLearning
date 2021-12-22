import src.models.model_baseline as model_baseline
import src.models.model_downstream as model_downstream
import src.models.model_custom as model_custom
import src.models.model_proposed01 as model_proposed01
import torch


def load_model(config, model_name, checkpoint_path=None):
    model = None
    if model_name == 'CPCModel':
        model = model_baseline.CPCModel(
            args=config,
            g_enc_input=config['g_enc_input'],
            g_enc_hidden=config['g_enc_hidden'],
            g_ar_hidden=config['g_ar_hidden'],
            filter_sizes=config['filter_sizes'],
            strides=config['strides'],
            paddings=config['paddings'],
        )
    elif model_name == 'GenerativeCPCModel':
        model = model_proposed01.GenerativeCPCModel(
            args=config,
            g_enc_input=config['g_enc_input'],
            g_enc_hidden=config['g_enc_hidden'],
            g_ar_hidden=config['g_ar_hidden'],
            filter_sizes=config['filter_sizes'],
            strides=config['strides'],
            paddings=config['paddings'],
        )
    elif model_name == 'SpeakerClassification':
        model = model_downstream.SpeakerClassification(
            hidden_dim=config['g_ar_hidden'],
            speaker_num=config['speaker_num']
        )
    elif model_name == "PhonemeClassification":
        model = model_downstream.PhonemeClassification(
            hidden_dim=config['g_ar_hidden'],
            speaker_num=config['num_classes']
        )
    elif model_name == 'ResnetCombineModel':
        model = model_custom.ResNetCombine(
            config=config,
            embedding_size=config['embedding_size'],
            num_classes=config['speaker_num'],
            backbone='resnet18',
            pretrained=False,
        )
    elif model_name == 'VGGCombineModel':
        model = model_custom.VGGCombine(
            config=config,
            embedding_size=config['embedding_size'],
            num_classes=config['speaker_num']
        )
    elif model_name == 'MobileNetV2CombineModel':
        model = model_custom.MobileNetV2Combine(
            config=config,
            embedding_size=config['embedding_size'],
            num_classes=config['speaker_num']
        )
    elif model_name == 'SqueezeNetCombineModel':
        model = model_custom.SqueezeNetCombine(
            config=config,
            embedding_size=config['embedding_size'],
            num_classes=config['speaker_num']
        )

    if checkpoint_path is not None:
        device = torch.device('cpu')
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])

    return model

