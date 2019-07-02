import json
import os


class ClassDict(dict):
    """
    Object like dict, every dict[key] can be visited by dict.key
    """

    def __getattr__(self, name):
        return self.__getitem__(name)

    def __setattr__(self, key, value):
        self.update({key: value})

    def write(self, path):
        path = os.path.splitext(path)[0]
        with open('%s.json' % path, 'w') as outfile:
            json.dump(self, outfile)

    def load(self, path):
        with open(path, 'r') as f:
            datastore = json.load(f)
            self.update(datastore)
        return self
