import glob
import multiprocessing
import subprocess
import tqdm
import os
from shutil import copyfile


"""
ERRORS: ['../data/txts/ACL19-50.pdf.txt', '../data/txts/NAACL19-437.pdf.txt', '../data/txts/EMNLP18-349.pdf.txt', '../data/txts/EMNLP19-268.pdf.txt', '../data/txts/COLING18-66.pdf.txt', '../data/txts/EMNLP19-16.pdf.txt', '../data/txts/ACL19-640.pdf.txt', '../data/txts/AAAI19-287.pdf.txt', '../data/txts/ACL19-405.pdf.txt', '../data/txts/NeurIPS18-390.pdf.txt', '../data/txts/EMNLP19-393.pdf.txt', '../data/txts/AAAI12-150.pdf.txt', '../data/txts/ACL19-658.pdf.txt', '../data/txts/ICLR19-438.pdf.txt', '../data/txts/AAAI18-439.pdf.txt', '../data/txts/ICML19-131.pdf.txt', '../data/txts/EMNLP18-189.pdf.txt', '../data/txts/NeurIPS18-671.pdf.txt', '../data/txts/AAAI15-459.pdf.txt', '../data/txts/AAAI18-286.pdf.txt', '../data/txts/EMNLP19-24.pdf.txt', '../data/txts/EMNLP18-132.pdf.txt', '../data/txts/AAAI19-790.pdf.txt', '../data/txts/ACL19-650.pdf.txt', '../data/txts/NeurIPS19-1171.pdf.txt', '../data/txts/ICML12-96.pdf.txt', '../data/txts/ICLR18-241.pdf.txt', '../data/txts/ACL18-45.pdf.txt', '../data/txts/EMNLP19-592.pdf.txt', '../data/txts/AAAI19-804.pdf.txt', '../data/txts/ICML18-430.pdf.txt', '../data/txts/ACL19-215.pdf.txt', '../data/txts/EMNLP19-28.pdf.txt', '../data/txts/AAAI19-719.pdf.txt', '../data/txts/EMNLP19-182.pdf.txt', '../data/txts/EMNLP18-210.pdf.txt', '../data/txts/ACL19-149.pdf.txt', '../data/txts/ACL19-49.pdf.txt', '../data/txts/AAAI19-686.pdf.txt', '../data/txts/ACL19-537.pdf.txt', '../data/txts/ACL19-655.pdf.txt', '../data/txts/ACL19-601.pdf.txt', '../data/txts/EMNLP18-451.pdf.txt', '../data/txts/EMNLP18-542.pdf.txt', '../data/txts/CoNLL19-8.pdf.txt', '../data/txts/ICML19-19.pdf.txt', '../data/txts/ACL19-359.pdf.txt', '../data/txts/AAAI19-22.pdf.txt', '../data/txts/ACL19-321.pdf.txt', '../data/txts/EMNLP19-18.pdf.txt', '../data/txts/NeurIPS19-552.pdf.txt', '../data/txts/NeurIPS18-309.pdf.txt', '../data/txts/ACL19-435.pdf.txt', '../data/txts/NeurIPS19-723.pdf.txt', '../data/txts/ICML19-316.pdf.txt', '../data/txts/ICLR19-391.pdf.txt']
"""


def process_file(file_path):
    new_path = f"{file_path.replace('pdfs', 'txts2')}.txt"
    if os.path.exists(new_path):
        return
    with open(new_path, "w") as file:
        code = subprocess.call(["pdf2txt.py", file_path], stdout=file)
    if code != 0:
        file_path = f"{file_path.replace('pdfs', 'txts')}.txt"
        copyfile(file_path, new_path)
        return file_path


if __name__ == "__main__":
    file_paths = glob.glob("../data/pdfs/*.pdf")



    errors = []
    with multiprocessing.Pool() as pool:
        for e in tqdm.tqdm(pool.imap_unordered(process_file, file_paths), total=len(file_paths)):
            errors.append(e)

    errors = [x for x in errors if x is not None]
    print(errors)
