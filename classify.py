import argparse
import logging
from os.path import exists, join, dirname
from pipeline import Pipeline, process_with_pool

parser = argparse.ArgumentParser()
parser.add_argument("-d", "--directory",
                    help="data directory to process, e.g. data/txts_cleaned/",
                    type=str)
parser.add_argument("-f", "--file",
                    help="single pdf file to process, e.g. data/txts_cleaned/AAAI12-0.txt",
                    type=str)
parser.add_argument("-gpu",
                    help="activate hardware acceleration (this may require up to 16GB of GPU memory, take care!)",
                    action="store_true")
parser.add_argument("-v", "--verbose",
                    help="print additional output (mainly for debugging)",
                    action="store_true")

args = parser.parse_args()

if not (args.directory or args.file):
    parser.print_help()

if args.verbose:
    logging.basicConfig(format='%(asctime)s %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO)
if args.gpu:
    device = "cuda"
else:
    device = "cpu"

if args.file:
    data_directory = dirname(args.file)
    assert exists(args.file), f"'{args.file}' does not exist"

    logging.info(f"loading pipeline for file '{args.file}'")
    pipeline = Pipeline.classification_factory(data_directory)
    document = pipeline(args.file)
    print("{0:<25}{1}".format("Name", "Prediction"))
    print("-" * 50)
    print(f"{document['name']:<25}{document['prediction']}")
    logging.info(f"finished processing '{args.file}'")

if args.directory:
    assert exists(args.directory), f"data directory '{args.directory}' does not exist"

    logging.info(f"spawning pool for directory '{args.directory}'")
    process_with_pool(args.directory,
                      pipeline=Pipeline.classification_factory,
                      processes=1,
                      device=device)
    logging.info(f"finished processing directory '{args.directory}'")

