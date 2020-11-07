# %% Setup
import os
import pandas as pd
import time
import datetime

import pytesseract
from pdf2image import convert_from_path

# %% Define PDF and Page Class


# %% Main Structure
def main():
    start = time.time()

    # i) specify cols for final df
    columns = ['text', 'pagenr', 'filename']  # TODO add date to df somehow?
    data = []

    pdf_dir = "data/testPDFs/"

    # ii) loop through pdf files
    for fname in os.listdir(pdf_dir):
        path = pdf_dir + fname

        imgs = convert_from_path(path)

        # iv) loop images and apply OCR
        for i, img in enumerate(imgs):

            # tesseract needs to be installed + german training data has to be downloaded and put into
            # C:\ProgramData\Anaconda3\envs\*env_name*\Library\bin\tessdata
            # can be downloaded from https://github.com/tesseract-ocr/tessdata/blob/master/deu.traineddata
            text = pytesseract.image_to_string(img, lang='deu')
            data.append([text, i, fname])

    # save final df
    df = pd.DataFrame(data, columns=columns)
    df.to_pickle(pdf_dir + 'articles_raw_gen{}.pkl'.format(datetime.date.today()))

    print("Finished! Saved PDFs as DataFrame. Elapsed time: {}".format(time.gmtime(time.time() - start)))


if __name__ == '__main__':
    main()
