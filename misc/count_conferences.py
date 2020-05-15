import os
import glob
import tqdm
import textract

file_paths = glob.glob("../data/pdfs/*.pdf")
file_paths = [file_path.replace("../data/pdfs/", "") for file_path in file_paths]
file_paths = [file_path.split("-")[0] for file_path in file_paths]

print(len(set(file_paths)))
file_paths = set(file_paths)

for conference_path in tqdm.tqdm(file_paths):
    for paper in glob.glob(f"../data/pdfs/{conference_path}-*.pdf"):
        new_path = f"{paper.replace('pdfs', 'txts_single_per_conference')}.txt"

        try:
            text = textract.process(paper)
            with open(new_path, "w") as file:
                file.write(text.decode('UTF-8'))
        except:
            continue

        break
