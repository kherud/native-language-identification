import abc
import re
from enum import Enum


class Entity(Enum):
    ACKNOWLEDGEMENT = "Acknowledgements"
    AUTHOR = "Autor"
    COUNTRY = "Country"
    EMAIL = "Email"
    FOOTNOTE = "Footnote"
    INSTITUTION_COMPANY = "Institution/Company"
    LANGUAGE = "Language"
    PERSONAL_DATA = "Personal Data"
    REFERENCE = "References"
    REVIEWER = "Reviewer"

    def __str__(self):
        return str(self.value)


class Target(abc.ABC):
    def __init__(self):
        pass

    @abc.abstractmethod
    def __call__(self, document):
        pass

    def clean_text(self, document, substring, cased=False):
        regex = "\s*".join(re.escape(c) for c in substring)
        if cased:
            document["text_cleaned"] = re.sub(regex, "", document["text_cleaned"])
        else:
            document["text_cleaned"] = re.sub(regex, "", document["text_cleaned"], re.IGNORECASE)


def worker(target: Target, pipe_in, pipe_out, lock):
    while True:
        with lock:
            try:
                if hasattr(pipe_in, "recv"):
                    value = pipe_in.recv()
                else:
                    value = pipe_in.get()
            except EOFError:
                break
        result = target(value)
        if pipe_out is not None:
            pipe_out.send(result)

