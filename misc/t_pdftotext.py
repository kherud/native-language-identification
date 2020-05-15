import pdftotext

file_name = "ACL12-50.pdf"

with open(f"../data/pdfs/{file_name}", "rb") as f:
    pdf = pdftotext.PDF(f)

with open(f"{file_name}.txt", "w") as f:
    for page in pdf:
        f.write(page)
