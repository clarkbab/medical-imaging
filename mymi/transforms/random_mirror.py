class RandomMirror:
    def __init__(self):
        return None

    def cache_key(self):
        raise ValueError("Random transformations aren't cacheable.")
