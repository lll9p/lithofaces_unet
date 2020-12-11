import yaml
from .defaults import defaults


class Config(dict):
    @staticmethod
    def from_yml(yml):
        with open(yml, mode="r") as file:
            config = yaml.load(file, Loader=yaml.FullLoader)
            Config.check_classes(config)
        return config

    @staticmethod
    def check_classes(config):
        labels_length = len(config["labels"])-len(config["ignore_labels"])+1
        # including background
        if config["num_classes"] != labels_length:
            config["num_classes"] = labels_length

    def __init__(self, defaults=defaults, yml=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.update(defaults)
        if yml is not None:
            config = Config.from_yml(yml)
            self.update(config)
        Config.check_classes(self)

    def from_dict(self, config_dict):
        self.update(config_dict)
        Config.check_classes(self)

    def save(self, yml):
        with open(yml, mode="w") as file:
            yaml.dump(self, file)

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


if __name__ == "__main__":
    print(Config())
