import re
import string
from typing import Union
from tokenizers.implementations import BaseTokenizer
from . import Target


class Tokenization(Target):
    def __init__(self, tokenizer: BaseTokenizer,
                 max_line_length: Union[int, None] = 50,
                 padding_id: int = 0):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_line_length = max_line_length
        self.padding_id = padding_id
        self.char_re = re.compile(rf"[^{string.printable}]")

    def __call__(self, document):
        assert isinstance(document, dict), f"wrong input of type {type(document)} to tokenizer"

        processed_text = self.char_re.sub("", document["text"])
        lines = [line for line in processed_text.split("\n") if len(line) > 0]
        encoded_lines = []
        for line in self.tokenizer.encode_batch(lines):
            if self.max_line_length is not None:
                encoded_lines.append(line.ids[:self.max_line_length])
            else:
                encoded_lines.append(line.ids)

        max_line_length = max(len(line) for line in encoded_lines)
        for j in range(len(encoded_lines)):
            amount_padding = max_line_length - len(encoded_lines[j])
            encoded_lines[j] = [self.padding_id] * amount_padding + encoded_lines[j]

        document["input_ids"] = encoded_lines

        return document


