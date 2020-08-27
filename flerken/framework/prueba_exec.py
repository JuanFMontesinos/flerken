import sys
class Experiment(object):
    def __init__(self):
        print('init')
        self.x=5
    def __enter__(self):
        print('enter')
        self.x=12
        print(self.x)

        return self
    def __exit__(self, exc_type, exc_val, exc_tb):
        if isinstance(exc_val,Exception):
            exc_val.args = (*exc_val.args,)
            raise
experimento = Experiment()
with experimento as ex:
    w=ex.x
    print(w)
