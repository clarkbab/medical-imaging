from collections import namedtuple
import os

class Directories:
    @property
    def cache(self):
        return os.path.join(self.root, 'cache')

    @property
    def checkpoint(self):
        return os.path.join(self.root, 'checkpoints')

    @property
    def dataset(self):
        return os.path.join(self.root, 'datasets')
    
    @property
    def evaluation(self):
        return os.path.join(self.root, 'evaluation')

    @property
    def root(self):
        return os.environ['MYMI_DATA']

    @property
    def tensorboard(self):
        return os.path.join(self.root, 'tensorboard')

class Formatting:
    @property
    def metrics(self):
        return '.6f'

    @property
    def sample_digits(self):
        return 5

directories = Directories()
formatting = Formatting()