import yaml
from .defaults import defaults


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __getattr__(self, name):
        if name in self:
            return self[name]
        else:
            raise AttributeError("No such attribute: " + name)

    def __setattr__(self, name, value):
        self[name] = value

    def __delattr__(self, name):
        if name in self:
            del self[name]
        else:
            raise AttributeError("No such attribute: " + name)


class Config(AttrDict):
    @staticmethod
    def from_yml(yml):
        with open(yml, mode="r") as file:
            config = yaml.load(file, Loader=yaml.FullLoader)
            Config.check_classes(config)
        return config

    @staticmethod
    def from_dict(config_dict):
        config = Config(defaults=config_dict)
        Config.check_classes(config)
        return config

    @staticmethod
    def check_classes(config):
        diff = len(config['dataset']["labels"]
                   ) - len(config['dataset']["ignore_labels"])
        if diff == 0:
            # on edge mode
            labels_length = diff + 2
        elif diff > 0:
            labels_length = diff + 1
        # including background
        if config['model']["num_classes"] != labels_length:
            config['model']["num_classes"] = labels_length

    def __init__(self, defaults=defaults, yml=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for key, value in defaults.items():
            key_ = key
            if isinstance(value, dict):
                value_ = AttrDict()
                for value_key, value_value in value.items():
                    if isinstance(value_value, dict):
                        value_temp = AttrDict()
                        value_temp.update(value_value)
                        value_[value_key] = value_temp
                    else:
                        value_[value_key] = value_value
            else:
                value_ = value
            self[key_] = value_
        if yml is not None:
            config = Config.from_yml(yml)
            self.update(config)
        Config.check_classes(self)

    def save(self, yml):
        with open(yml, mode="w") as file:
            yaml.dump(self, file)


if __name__ == "__main__":
    print(Config())
