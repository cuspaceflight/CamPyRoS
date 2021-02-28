import warnings


class remote:
    def __init__(self, f):
        warnings.warn(
            "Ray was not available so multithread running will not work. Stats will take a long time"
        )
        self.f = f

    def __call__(self):
        self.f
