import glob
from pipeline import PipelineMultiprocess, PipelineSingleprocess, process_with_pool

# > spacy download en_core_web_sm

data_directory = "data"

# files are in data/pdfs and data/txts
# pipeline = PipelineMultiprocess.factory(data_directory)
# pipeline.start()

# pipeline = PipelineSingleprocess.factory(data_directory)
# print(pipeline("data/pdfs/AAAI12-4.pdf"))

process_with_pool(data_directory)

# for x in


