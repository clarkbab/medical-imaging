import yaml

from .. import transforms

class TransformParser:
    @staticmethod
    def __call__(path):
        """
        returns: a list of parsed transforms.
        """
        # Load yaml data.
        with open(path, 'r') as s:
            data = yaml.safe_load(s)

        assert isinstance(data, list)

        return [TransformParser.parse_transform(d) for d in data]

    @staticmethod
    def parse_transform(data):
        """
        returns a transform.
        data: transform data representation.
        """
        # Get transform name.
        name = data['name']
        assert name is not None

        if name == 'random_rotation':
            return TransformParser.parse_random_rotation(data)
        elif name == 'random_translation':
            return TransformParser.parse_random_translation(data)
        else:
            raise ValueError

    @staticmethod
    def parse_random_rotation(data):
        """
        returns: random rotation transform.
        data: required data.
        """
        # Get transform params.
        fill = data['fill']
        range = data['range']

        return transforms.RandomRotation(range, fill)

    @staticmethod
    def parse_random_translation(data):
        """
        returns: random translation transform.
        data: required data.
        """
        # Get transform params.
        fill = data['fill']
        x_range = data['x-range']
        y_range = data['y-range']

        return transforms.RandomTranslation((x_range, y_range), fill)