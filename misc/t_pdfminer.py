import subprocess

file_name = "ACL12-50.pdf"

with open(f"{file_name}.txt", "wb") as file:
    code = subprocess.call(["pdf2txt.py", f"../data/pdfs/{file_name}"], stdout=file)