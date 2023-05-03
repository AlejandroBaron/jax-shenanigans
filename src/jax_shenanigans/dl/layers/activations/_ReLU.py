from ._LeakyReLU import LeakyReLU


class ReLU(LeakyReLU):
    def __init__(self) -> None:
        super().__init__(alpha=0)
