from pipeline import Pipeline

# > spacy download en_core_web_sm

# files are in data/pdfs and data/txts
pipeline = Pipeline.factory("data")
pipeline.start()
