import yaml


def load_config(path):
    with open(path) as f:
        return yaml.load(f,Loader=yaml.FullLoader)


if __name__ =='__main__':
    print(load_config('config.yaml'))