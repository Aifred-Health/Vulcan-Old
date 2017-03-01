import math

class Sigmoid:
    def apply(self, val):
        try:
            return (1/(1 + math.exp(val)))
        except OverflowError:
            return 0

    def derivative(self, val):
        return (val * (1 - val))
