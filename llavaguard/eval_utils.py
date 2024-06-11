import json
import os


def get_model_dir(run_name):
    if os.path.exists(run_name):
        return run_name
    if os.path.exists(f'/common-repos/LlavaGuard/models/{run_name}'):
        return f'/common-repos/LlavaGuard/models/{run_name}'
    elif os.path.exists(f'output/models/{run_name}'):
        return f'output/models/{run_name}'
    else:
        return None


def load_data(data_path, split='eval'):
    dd = {}
    paths = {}
    if data_path.endswith('.json'):
        dd = {data_path.split('/')[-1].split('.')[0]: json.load(open(data_path))}
        paths = {data_path.split('/')[-1].split('.')[0]: data_path}
        return paths, dd
    split = [split] if isinstance(split, str) else split
    data = [(data_path, s) for s in split]
    for p, type in data:
        # if type == 'train' and not infer_train_data:
        #     continue
        if not p.endswith('/'):
            p += '/'
        p += f'{type}.json'
        if os.path.exists(p):
            dd[type] = json.load(open(p))
        elif os.path.exists(f'/common-repos/LlavaGuard/data/{p}'):
            dd[type] = json.load(open(f'/common-repos/LlavaGuard/data/{p}'))
        elif os.path.exists(f'output/data/{p}'):
            dd[type] = json.load(open(f'output/data/{p}'))
        else:
            raise FileNotFoundError(f'No data found for {p}')
        paths[type] = p
    return paths, dd

