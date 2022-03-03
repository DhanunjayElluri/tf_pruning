import json


class Config:
    def __init__(self, data, model, prune):
        self.data = data
        self.model = model
        self.prune = prune

    @classmethod
    def from_json(cls, config):
        params = json.loads(json.dumps(config), object_hook=HelperObject)
        return cls(params.data, params.model, params.prune)


class HelperObject(object):
    def __init__(self, _dict):
        self.__dict__.update(_dict)
