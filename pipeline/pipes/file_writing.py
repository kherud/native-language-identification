import os
import pandas as pd
from . import Target

NER_LABEL_MAP = {
    "PER": "Autor"
    # ...
}

class CsvWriter(Target):
    def __init__(self, data_directory):
        super().__init__()
        self.output_path = os.path.join(data_directory, "csv")

        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path)

    def __call__(self, document):
        assert type(document) == dict, f"input to file parser has wrong type: {type(document)}"
        assert "name" in document, "document not yet parsed"
        assert "entities" in document, "document not yet processed"

        result = {
            "Text": [],
            "Label": [],
        }

        for entity_name, entities in document["entities"].items():
            if entity_name in NER_LABEL_MAP:
                entity_name = NER_LABEL_MAP[entity_name]
            for entity in entities:
                result["Text"].append(entity)
                result["Label"].append(entity_name)

        file_path = os.path.join(self.output_path, document["name"] + ".csv")

        pd.DataFrame(result).to_csv(file_path, index=False)

        return result