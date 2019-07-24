import json

config = {}

config['training_dir'] = '/mnt/data/train'
config['validation_dir'] = '/mnt/data/validation'
config['batch_size'] = 1
config['img_width_height'] = (224, 224)

config['model_parameters'] = {}
config['model_parameters']['mlp_dropout_rate'] = 0.5
config['model_parameters']['mlp_learning_rate'] = 0.0001


with open('config.json', 'w') as f:
    json.dump(config, f, indent=4)
