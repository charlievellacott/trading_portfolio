class Strategy:
    def __init__(self) -> None:
        pass

    def generate_features(self):
        raise NotImplementedError

    def generate_signal(self):
        raise NotImplementedError

    def get_weights(self):
        raise NotImplementedError