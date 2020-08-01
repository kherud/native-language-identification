# Installation

```
(git lfs pull)
conda env create --file=environment.yml
conda activate prjak
```

## Run

### Preprocessing

Loading the pipeline may take a while before processing begins.

```
clean_text.py [-h] [-d DIRECTORY] [-f FILE] [-p PROCESSES] [-gpu] [-v]

optional arguments:
  -h, --help            show this help message and exit
  -d DIRECTORY, --directory DIRECTORY
                        data directory to process, e.g. data/txts/
  -f FILE, --file FILE  single text file to process, e.g. data/txts/AAAI12-4.txt
  -p PROCESSES, --processes PROCESSES
                        amount of cpu cores used (defaults to all available)
  -gpu                  activate hardware acceleration (each process takes ~700mb GPU memory, take care!)
  -v, --verbose         print additional output (mainly for debugging)
```

### Classification

```
classify.py [-h] [-d DIRECTORY] [-f FILE] [-gpu] [-v]

optional arguments:
  -h, --help            show this help message and exit
  -d DIRECTORY, --directory DIRECTORY
                        data directory to process, e.g. data/txts_cleaned/
  -f FILE, --file FILE  single pdf file to process, e.g. data/txts_cleaned/AAAI12-0.txt
  -gpu                  activate hardware acceleration (this may require up to 16GB of GPU memory, take care!)
  -v, --verbose         print additional output (mainly for debugging)
```

## Example Usage

```
# process every file in data/txts/ cpu-based with all available cores
python clean_text.py -d data/txts/

# process every file in data/txts/ hardware-accelerated with four cores
python clean_text.py -d data/txts/ -p 4 -gpu

# process single file with hardware acceleration
python clean_text.py -f data/txts/AAAI12-4.txt -gpu

# classify single file
python classify.py -f data/txts_cleaned/AAAI12-4.txt

# classify every file in data/txts_cleaned using hardware acceleration
python classify.py -d data/txts_cleaned -gpu
```
