import src.models.model_cpc as model_baseline
import src.models.model_downstream as model_downstream
import src.models.model_proposed01 as model_proposed01
import src.models.model_proposed02 as model_proposed02
import src.models.model_proposed03 as model_proposed03
import src.models.model_proposed04 as model_proposed04
import src.models.model_proposed05 as model_proposed05
import src.models.model_proposed06 as model_proposed06
import src.models.model_proposed07 as model_proposed07
import src.models.model_proposed_efficientnet_combine as model_proposed_efficientnet_combine
import src.models.model_byol_audio as byol_audio
import torch


def load_model(config, model_name, checkpoint_path=None):
    model = None

    if model_name == "NormalClassification":
        model = model_downstream.DownstreamClassification(
            input_dim=config['downstream_input_dim'],
            hidden_dim=config['downstream_hidden_dim'],
            output_dim=config['downstream_output_dim']
        )
    # implemented models
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
    elif model_name == 'BYOLAudioModel':
        model = byol_audio.BYOLAudio(
            config=config,
            input_dims=config['input_dims'],
            hidden_dims=config['hidden_dims'],
            strides=config['strides'],
            filter_sizes=config['filter_sizes'],
            paddings=config['paddings'],
            maxpool_filter_sizes=config['maxpool_filter_sizes'],
            maxpool_strides=config['maxpool_strides'],
            feature_dimension=config['feature_dimension'],
            hidden_size=config['hidden_size'],
            projection_size=config['projection_size']
        )
    # proposed model
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
    elif model_name == 'WaveBYOLEfficientB4':
        model = model_proposed_efficientnet_combine.WaveBYOLEfficient(
            config=config,
            pre_input_dims=config['pre_input_dims'],
            pre_hidden_dims=config['pre_hidden_dims'],
            pre_strides=config['pre_strides'],
            pre_filter_sizes=config['pre_filter_sizes'],
            pre_paddings=config['pre_paddings'],
            dimension=config['dimension'],
            hidden_size=config['hidden_size'],
            projection_size=config['projection_size'],
            efficientnet_model_name='efficientnet-b4',
        )
    elif model_name == 'WaveBYOLEfficientB7':
        model = model_proposed_efficientnet_combine.WaveBYOLEfficient(
            config=config,
            pre_input_dims=config['pre_input_dims'],
            pre_hidden_dims=config['pre_hidden_dims'],
            pre_strides=config['pre_strides'],
            pre_filter_sizes=config['pre_filter_sizes'],
            pre_paddings=config['pre_paddings'],
            dimension=config['dimension'],
            hidden_size=config['hidden_size'],
            projection_size=config['projection_size'],
            efficientnet_model_name='efficientnet-b7',
        )
    elif model_name == 'WaveBYOL':
        model = model_proposed02.WaveBYOL(
            config=config,
            pre_input_dims=config['pre_input_dims'],
            pre_hidden_dims=config['pre_hidden_dims'],
            pre_strides=config['pre_strides'],
            pre_filter_sizes=config['pre_filter_sizes'],
            pre_paddings=config['pre_paddings'],
            dimension=config['dimension'],
            hidden_size=config['hidden_size'],
            projection_size=config['projection_size'],
        )
        model.setup_target_network()

    elif model_name == 'WaveBYOLTest01':
        model = model_proposed03.WaveBYOLTest01(
            config=config,
            pre_input_dims=config['pre_input_dims'],
            pre_hidden_dims=config['pre_hidden_dims'],
            pre_strides=config['pre_strides'],
            pre_filter_sizes=config['pre_filter_sizes'],
            pre_paddings=config['pre_paddings'],
            dimension=config['dimension'],
            hidden_size=config['hidden_size'],
            projection_size=config['projection_size'],
        )
        model.setup_target_network()

    elif model_name == 'WaveBYOLTest02':
        model = model_proposed04.WaveBYOLTest02(
            config=config,
            pre_input_dims=config['pre_input_dims'],
            pre_hidden_dims=config['pre_hidden_dims'],
            pre_strides=config['pre_strides'],
            pre_filter_sizes=config['pre_filter_sizes'],
            pre_paddings=config['pre_paddings'],
            dimension=config['dimension'],
            hidden_size=config['hidden_size'],
            projection_size=config['projection_size'],
        )
        model.setup_target_network()

    elif model_name == 'WaveBYOLTest03':
        model = model_proposed05.WaveBYOLTest03(
            config=config,
            pre_input_dims=config['pre_input_dims'],
            pre_hidden_dims=config['pre_hidden_dims'],
            pre_strides=config['pre_strides'],
            pre_filter_sizes=config['pre_filter_sizes'],
            pre_paddings=config['pre_paddings'],
            dimension=config['dimension'],
            hidden_size=config['hidden_size'],
            projection_size=config['projection_size'],
        )
        model.setup_target_network()
    elif model_name == 'EfficientBYOL':
        model = model_proposed07.EfficientBYOL(
            config=config,
            pre_input_dims=config['pre_input_dims'],
            pre_hidden_dims=config['pre_hidden_dims'],
            pre_strides=config['pre_strides'],
            pre_filter_sizes=config['pre_filter_sizes'],
            pre_paddings=config['pre_paddings'],
            dimension=config['dimension'],
            hidden_size=config['hidden_size'],
            projection_size=config['projection_size'],
        )
        model.setup_target_network()
        model.setup_modest_network()

    if checkpoint_path is not None:
        device = torch.device('cpu')
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)

    return model

