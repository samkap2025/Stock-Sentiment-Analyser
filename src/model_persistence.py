import pickle

def save_models(models, path='models/'):
    for model_name, model in models.items():
        filepath = f'{path}{model_name}_model.pkl'
        with open(filepath, 'wb') as f:
            pickle.dump(model, f)
        print(f'Saved {model_name} to {filepath}')


def load_models(path='models/'):
    models = {}
    import os

    for filename in os.listdir(path):
        if filename.endswith('.pkl'):
            model_name = filename.replace('_model.pkl', '')
            with open(f'{path}{filename}', 'rb') as f:
                models[model_name] = pickle.load(f)

    return models


def save_scaler(scaler, path='models/scaler.pkl'):
    with open(path, 'wb') as f:
        pickle.dump(scaler, f)

def load_scaler(path='models/scaler.pkl'):
    with open(path, 'rb') as f:
        return pickle.load(f)

