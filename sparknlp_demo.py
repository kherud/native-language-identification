from sparknlp.base import *
from sparknlp.annotator import *

from sparknlp.pretrained import PretrainedPipeline
import sparknlp

# Start Spark Session with Spark NLP
spark = sparknlp.start()

# Download a pre-trained pipeline
# pipeline = PretrainedPipeline('recognize_entities_bert', lang='en')

# Your testing dataset
text = """The Mona Lisa is a 16th century oil painting created by Leonardo.
It's held at the Louvre in Paris."""

data = spark.createDataFrame([
    ["The Mona Lisa is a 16th century oil painting created by Leonardo."],
    ["It's held at the Louvre in Paris."]
], ["text"])

# df = spark.createDataFrame(text, ["Sentences"])
# print(pipeline.annotate(df, "Sentences"))

# Annotate your testing dataset
# result = pipeline.annotate(text)

# What's in the pipeline
# list(result.keys())
#
# Check the results
# for k in result:
#     print(k, result[k])

documentAssembler = DocumentAssembler()\
    .setInputCol("text")\
    .setOutputCol("document")

sentenceDetector = SentenceDetector()\
    .setInputCols(["document"])\
    .setOutputCol("sentence")

tokenizer = Tokenizer()\
    .setInputCols(["sentence"])\
    .setOutputCol("token")

normalizer = Normalizer()\
    .setInputCols("token")\
    .setOutputCol("normal")

bertEmbeddings = BertEmbeddings\
    .pretrained("bert_base_cased")\
    .setInputCols(["sentence", "token"])\
    .setOutputCol("bertEmbd")\
    .setCaseSensitive(True)

nerBertModel = NerDLModel\
    .pretrained("ner_dl_bert")\
    .setInputCols(["sentence", "token", "bertEmbd"])\
    .setOutputCol("nerBert")

nerConverter = NerConverter()\
    .setInputCols(["document", "token", "nerBert"])\
    .setOutputCol("ner_converter")

finisher = Finisher()\
    .setInputCols("nerBert", "ner_converter")\
    .setIncludeMetadata(False)\
    .setOutputAsArray(True)\
    .setCleanAnnotations(False)\
    .setAnnotationSplitSymbol("@")\
    .setValueSplitSymbol("#")

pipeline = Pipeline(stages=[
    documentAssembler,
    sentenceDetector,
    tokenizer,
    bertEmbeddings,
    nerBertModel,
    nerConverter,
    finisher
])

print(dir(pipeline))
print(pipeline)
model = pipeline.fit(data)

result = model.transform(data).toPandas()
result.to_csv("result.csv")
print(result)
# print(result.finished_ner_converter.values)
# print(result.finished_nerBert.values)
# for k in result:
#     print(result[k])

# result = pipeline.fit([]).transform(text)

# result.printSchema()

