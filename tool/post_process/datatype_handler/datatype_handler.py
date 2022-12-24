import abc
import os
from PIL import Image

class Data_type_handler(metaclass=abc.ABCMeta):
    def __init__(self,args):
        self.args=args
        self.process_file = None

    @abc.abstractmethod
    def get_item(self, f):
        return NotImplemented

    @abc.abstractmethod
    def resize(self, f):
        return NotImplemented
    
    @abc.abstractmethod
    def store_file(self, f):
        return NotImplemented