from abc import ABCMeta

'''
import functools

# Decorator fonction to create singleton Class
# source: https://realpython.com/primer-on-python-decorators/
# Usage:
# @singleton
# class TheOne:
#    pass
def singleton(cls):
    """ Make a class a Singleton class (only one instance). """

    @functools.wraps(cls)
    def wrapper_singleton(*args, **kwargs):
        if not wrapper_singleton.instance:
            wrapper_singleton.instance = cls(*args, **kwargs)
        return wrapper_singleton.instance

    wrapper_singleton.instance = None
    return wrapper_singleton
'''


# source: https://stackoverflow.com/questions/33364070/implementing-singleton-as-metaclass-but-for-abstract-classes
class SingletonABCMeta(ABCMeta):

    _instances = {}
    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(SingletonABCMeta, cls).__call__(*args, **kwargs)
        return cls._instances[cls]




# Prevent Overriding of methods by derived classes?:
# see discussion here: https://stackoverflow.com/questions/3948873/prevent-function-overriding-in-python
