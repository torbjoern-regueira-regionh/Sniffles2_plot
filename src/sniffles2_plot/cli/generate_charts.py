# -*- coding: utf-8 -*-
"""
Created on Mon May 15 10:17:54 2023

@author: HGSC - Farhang Jaryani
"""
from sniffles2_plot.cli import generate_multi_vcf_charts, single_visulaizer
from sniffles2_plot.helper.io_class import open_vcf

HEADER_SIGN = "#"


def _is_multi_vcf(input_vcf_file_path: str) -> bool:
    """Return True if the VCF has more than one sample column."""
    with open_vcf(input_vcf_file_path) as f:
        for line in f:
            if line.startswith("#CHROM") or line.startswith("#chrom"):
                return len(line.split("\t")) > 10  # 9 fixed cols + >1 sample
    return False


def generate_charts(input_vcf_file_path: str, output_directory_path: str) -> None:
    if _is_multi_vcf(input_vcf_file_path):
        generate_multi_vcf_charts(input_vcf_file_path, output_directory_path)
    else:
        single_visulaizer(input_vcf_file_path, output_directory_path)
