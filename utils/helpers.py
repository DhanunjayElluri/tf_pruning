import os
import tempfile
import zipfile

class Helpers:
    def __init__(self, file_name):
        self.file_name = file_name

    def get_gzipped_model(self):
        _, zipped_file = tempfile.mkstemp('.zip')
        with zipfile.ZipFile(zipped_file, 'w', compression=zipfile.ZIP_DEFLATED) as f:
            f.write(self.file_name)
        return os.path.getsize(zipped_file)