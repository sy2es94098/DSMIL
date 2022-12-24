from datatype_handler import Datatype_handler
import openslide
Image.MAX_IMAGE_PIXELS=None

class Tiff_handler(Pretrain_model):
    def __init__(self, args):
        super().__init__(args)

    def get_item(self, f):
        self.process_file = openslide.open_slide(os.path.join(self.args.dir,f))
        return self.process_file
    
    def resize(self, f):
        w, h = self.process_file.level_dimensions[args.level]
        self.slide = self.process_file.read_region((0, 0),zoom_level, (w, h))
        return self.slide

    def store_file(self, f):
        self.slide.save(os.path.join(self.args.store_dir,f))
