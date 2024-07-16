import pdfplumber
import os

filepath = "/Users/andreas/Desktop/DCLH_Research/soci_texts/comte/presentation_of_self.pdf"

def pdftotext(file):
    file_name = os.path.basename(file)
    with open(f"{file_name}.txt", 'w', encoding='utf8') as output:
            with pdfplumber.open(file) as pdf_file:
                  for pages in pdf_file.pages:
                        text = pages.extract_text()
                        output.write(f"{text}\n\n")

            return output, file_name 
