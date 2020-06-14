# Installation

```
(git lfs pull)
conda env create --file=environment.yml
conda activate prjak
```

## Run

Move data to `data/pdfs` and `data/txts`.
Loading the pipeline may take a while before processing begins.

```
extract_entities.py [-h] [-d DIRECTORY] [-f FILE] [-p PROCESSES] [-gpu] [-v]

optional arguments:
  -h, --help            show this help message and exit
  -d DIRECTORY, --directory DIRECTORY
                        data directory to process, e.g. data/ (expected to
                        contain pdfs/ and txts/)
  -f FILE, --file FILE  single pdf file to process, e.g.
                        data/pdfs/AAAI12-4.pdf
  -p PROCESSES, --processes PROCESSES
                        amount of cpu cores used (defaults to all available)
  -gpu                  activate hardware acceleration (each process takes 
                        ~700mb GPU memory, take care!)
  -v, --verbose         print additional output (mainly for debugging)
```

### Example

Make sure to put your files under `<data_directory>/pdfs` and `<data_directory>/txts` if you want to process a directory.

```
# process every file in data/ cpu-based with all available cores
python extract_entities.py -d data/

# process every file in data/ hardware-accelerated with four cores
python extract_entities.py -d data/ -p 4 -gpu

# process single file with hardware acceleration
python extract_entities.py -f data/pdfs/AAAI12-4.pdf -gpu
```

### In Code
Use `PipelineSingleprocess` for single files:

```
pipeline = PipelineSingleprocess.factory(data_directory)
# custom pipeline
pipeline = PipelineSingleprocess(pipeline=[
                                           # ...
                                           ])
print(pipeline(file_path))
```

To process a whole directory there is a wrapper `process_with_pool` to spawn a worker pool:

```
process_with_pool(data_directory)
# custom pipeline
process_with_pool(data_directory, pipeline=[
                                            # ...
                                            ])
```