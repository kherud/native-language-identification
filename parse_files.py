import argparse
from os.path import exists, join, dirname
from pipeline import PipelineSingleprocess, process_with_pool
import logging

parser = argparse.ArgumentParser()
parser.add_argument("-d", "--directory",
                    help="data directory to process, e.g. data/ (expected to contain pdfs/ and txts/)",
                    type=str)
parser.add_argument("-f", "--file",
                    help="single pdf file to process, e.g. data/pdfs/AAAI12-4.pdf",
                    type=str)
parser.add_argument("-p", "--processes",
                    help="amount of cpu cores used (defaults to all available)",
                    type=int)
parser.add_argument("-v", "--verbose",
                    help="print additional output (mainly for debugging)",
                    type=int)

args = parser.parse_args()

if not (args.directory or args.file):
    parser.print_help()

if args.verbose:
    logging.basicConfig(format='%(asctime)s %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO)

if args.directory:
    assert exists(args.directory), f"data directory '{args.directory}' does not exist"
    assert exists(join(args.directory, "pdfs")), f"pdfs/ directory not found in '{args.directory}'"
    assert exists(join(args.directory, "txts")), f"txts/ directory not found in '{args.directory}'"

    logging.info("spawning pool")
    process_with_pool(args.directory, processes=args.processes)
    logging.info("finished processing")

if args.file:
    assert exists(args.file), f"pdf file '{args.file}' does not exist"
    data_directory = dirname(dirname(args.file))
    assert exists(args.file.replace("pdfs", "txts") + ".txt"), f"associated txt file for '{args.file}' does not exist in '{join(data_directory, 'txts')}'"

    logging.info("loading pipeline")
    pipeline = PipelineSingleprocess.factory(data_directory)
    result = pipeline("data/pdfs/AAAI12-4.pdf")
    print("{0:<25}{1}".format("Label", "Text"))
    print("-" * 50)
    for key, value in zip(result["Label"], result["Text"]):
        print(f"{key:<25}{value}")
    logging.info("finished processing")
