import src.models.model_baseline as model_baseline


def load_model(config):
    model_name = config['model_type']
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
    return model

