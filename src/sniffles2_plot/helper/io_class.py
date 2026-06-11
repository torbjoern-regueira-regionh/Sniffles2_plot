import gzip
import os


def open_vcf(path: str):
    """Open a VCF file, transparently handling .gz compression."""
    path = str(path)
    if path.endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8")
    return open(path, "r", encoding="utf-8")


class FileIO:
    def __init__(self, input_file_path, output_directory):
        self.input_file_path = str(input_file_path)
        self.output_directory = output_directory

    def output_file(self, filename):
        """returnthe output file name and filepath"""
        return os.path.join(self.output_directory, filename)

    def open_input(self):
        """Open the input VCF file, transparently handling .gz compression."""
        if self.input_file_path.endswith(".gz"):
            return gzip.open(self.input_file_path, "rt", encoding="utf-8")
        return open(self.input_file_path, "r", encoding="utf-8")
