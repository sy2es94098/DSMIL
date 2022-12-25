from datatype_handler.datatype_handler import Datatype_handler
from PIL import Image
import os
Image.MAX_IMAGE_PIXELS=None

class Png_handler(Datatype_handler):
    def __init__(self, args):
        super().__init__(args)

    def get_item(self, f):
        self.process_file = Image.open(os.path.join(self.args.dir,f))
        return self.process_file

    def resize(self, f):
        self.sample_file = Image.open(os.path.join(self.args.sample_dir,f))
        w, h = self.sample_file.size
        self.process_file.resize((w, h))
        return self.process_file
        
    def store_file(self, f):
        self.process_file.save(os.path.join(self.args.store_dir,f))
