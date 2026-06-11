import argparse
import os
from typing import Tuple
from sniffles2_plot.cli.generate_charts import generate_charts
from sniffles2_plot.cli.caller_comparison import generate_caller_comparison_upset


def _parse_arguments():
    parser = argparse.ArgumentParser(description="Sniffles2 plot generator")
    parser.add_argument(
        "-i", "--input",
        help="Path to a single VCF file or a directory of VCF files",
        required=True,
    )
    parser.add_argument(
        "-o", "--output",
        help="Output directory path",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help=(
            "Caller-comparison mode: treat each VCF in the input directory as "
            "a separate caller and generate upset plots showing call overlap."
        ),
    )
    parser.add_argument(
        "--slop",
        type=int,
        default=500,
        help="Window size in bp for matching SVs across callers (default: 500)",
    )
    parser.add_argument(
        "--min-callers",
        type=int,
        default=None,
        metavar="N",
        help=(
            "In --compare mode: report variants found by >= N callers. "
            "Writes a TSV, embeds a table on the plot (if <= --table-limit rows), "
            "or generates a genome scatter plot (if more rows)."
        ),
    )
    parser.add_argument(
        "--table-limit",
        type=int,
        default=50,
        metavar="N",
        help="Max rows to embed as a table on the upset plot (default: 50). "
             "Above this, a genome scatter plot is generated instead.",
    )
    args = parser.parse_args()
    return args.input, args.output, args.compare, args.slop, args.min_callers, args.table_limit


def _ensure_output_directory_exists(path:str) -> None:
        if os.path.exists(path) and not os.path.isdir(path):
            raise IOError("The given path is not a directory.")
        if not os.path.exists(path):
            os.mkdir(path)


def entry_point():
    input_file_path, output_directory_path, compare_mode, slop, min_callers, table_limit = _parse_arguments()

    if compare_mode:
        if os.path.isfile(input_file_path):
            raise ValueError("--compare requires a directory of VCF files, not a single file.")
        _ensure_output_directory_exists(output_directory_path)
        generate_caller_comparison_upset(
            input_file_path, output_directory_path,
            slop=slop, min_callers=min_callers, table_limit=table_limit,
        )
        return

    if os.path.isfile(input_file_path):
        _ensure_output_directory_exists(output_directory_path)
        generate_charts(input_file_path, output_directory_path)
    else:
        for file_name in os.scandir(input_file_path):
            lower = file_name.name.lower()
            if lower.endswith(".vcf") or lower.endswith(".vcf.gz"):
                directory_path = os.path.join(input_file_path, os.path.splitext(file_name.name)[0])
                os.makedirs(directory_path, exist_ok=True)
                generate_charts(file_name.path, directory_path)
                