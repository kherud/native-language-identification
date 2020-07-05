import argparse
import logging
from os.path import exists, join, dirname
from pipeline import PipelineSingleprocess, process_with_pool

parser = argparse.ArgumentParser()
parser.add_argument("-d", "--directory",
                    help="data directory to process, e.g. data/txts/",
                    type=str)
parser.add_argument("-f", "--file",
                    help="single pdf file to process, e.g. data/txts/AAAI12-4.pdf.txt",
                    type=str)
parser.add_argument("-p", "--processes",
                    help="amount of cpu cores used (defaults to all available)",
                    type=int)
parser.add_argument("-gpu",
                    help="activate hardware acceleration (each process takes ~700mb GPU memory, take care!)",
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

if args.directory:
    assert exists(args.directory), f"data directory '{args.directory}' does not exist"

    logging.info("spawning pool")
    process_with_pool(args.directory, processes=args.processes, device=device)
    logging.info("finished processing")

if args.file:
    data_directory = dirname(args.file)
    assert exists(args.file), f"'{args.file}' does not exist"

    logging.info("loading pipeline")
    pipeline = PipelineSingleprocess.factory(data_directory)
    document = pipeline(args.file)
    print("{0:<25}{1}".format("Label", "Text"))
    print("-" * 50)
    for key, value in zip(document["result"]["Label"], document["result"]["Text"]):
        print(f"{key:<25}{value}")
    logging.info("finished processing")
