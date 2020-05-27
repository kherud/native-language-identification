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
extract_entities.py [-d DIRECTORY] [-f FILE] [-p PROCESSES]

optional arguments:
  -d DIRECTORY, --directory DIRECTORY
                        data directory to process, e.g. data/ (expected to
                        contain pdfs/ and txts/)
  -f FILE, --file FILE  single pdf file to process, e.g.
                        data/pdfs/AAAI12-4.pdf
  -p PROCESSES, --processes PROCESSES
                        amount of cpu cores used (defaults to all available)
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