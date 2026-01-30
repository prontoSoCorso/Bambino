import config_moment as cfg   


def import_moment_model_classifier():
    from momentfm import MOMENTPipeline

    model = MOMENTPipeline.from_pretrained(
        "AutonLab/MOMENT-1-large", 
        model_kwargs={
            'task_name': 'classification',
            'n_channels': cfg.n_channels,
            'num_class': cfg.num_classes
        },
    )
    return model


def import_moment_model_embeddings():
    from momentfm import MOMENTPipeline
    
    model = MOMENTPipeline.from_pretrained(
        "AutonLab/MOMENT-1-large", 
        model_kwargs={'task_name': 'embedding'},
    )
    return model




