"""Freeze a class, i.e., prevent setting new attributes outside __init__.

Raises:
    TypeError: Raised when setting a new attribute in a frozen class.
"""


class FrozenClass(object):
    __isfrozen = False

    def __setattr__(self, key, value):
        if self.__isfrozen and not hasattr(self, key):
            raise TypeError("%r is a frozen class" % self)
        object.__setattr__(self, key, value)

    def _freeze(self):
        self.__isfrozen = True
