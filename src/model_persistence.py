import pickle
import os


def save_models(models, path='models/'):
    os.makedirs(path, exist_ok=True)

    for model_name, model in models.items():
        filepath = f'{path}{model_name}_model.pkl'
        try:
            with open(filepath, 'wb') as f:
                pickle.dump(model, f)
            print(f'✓ Saved {model_name} to {filepath}')
        except Exception as e:
            print(f'✗ Error saving {model_name}: {e}')


def load_models(path='models/'):
    models = {}

    if not os.path.exists(path):
        print(f"Models directory not found: {path}")
        return models

    for filename in os.listdir(path):
        if filename.endswith('.pkl') and 'model' in filename:
            model_name = filename.replace('_model.pkl', '')
            try:
                with open(f'{path}{filename}', 'rb') as f:
                    models[model_name] = pickle.load(f)
                print(f'✓ Loaded {model_name}')
            except Exception as e:
                print(f'✗ Error loading {model_name}: {e}')

    return models


def save_scaler(scaler, path='models/scaler.pkl'):
    os.makedirs(os.path.dirname(path), exist_ok=True)

    try:
        with open(path, 'wb') as f:
            pickle.dump(scaler, f)
        print(f'✓ Saved scaler to {path}')
    except Exception as e:
        print(f'✗ Error saving scaler: {e}')


def load_scaler(path='models/scaler.pkl'):
    if not os.path.exists(path):
        print(f"Scaler file not found: {path}")
        return None

    try:
        with open(path, 'rb') as f:
            scaler = pickle.load(f)
        print(f'✓ Loaded scaler from {path}')
        return scaler
    except Exception as e:
        print(f'✗ Error loading scaler: {e}')
        return None