import textract

file_name = "ACL12-50.pdf"

text = textract.process(f"../data/pdfs/{file_name}")

with open(f"{file_name}.txt", "w") as file:
    file.write(text.decode('UTF-8'))