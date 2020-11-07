# %% Setup
import os
import pandas as pd
import time
import datetime

import pytesseract
from pdf2image import convert_from_path


# %% Main
def main():
    start = time.time()

    # i) specify cols for final df
    columns = ['date', 'text', 'pagenr', 'filename']
    data = []

    pdf_dir = "data/lagebeurteilungenSNB/"

    # ii) loop through pdf files
    for i, fname in enumerate(os.listdir(pdf_dir)):
        print("Extracting text from '{}' ({}/{})".format(fname, i+1, len(os.listdir(pdf_dir))))
        path = pdf_dir + fname
        imgs = convert_from_path(path)

        # iii) loop page-images and apply OCR
        for i, img in enumerate(imgs):
            # tesseract needs to be installed + german training data has to be downloaded and put into
            # C:\ProgramData\Anaconda3\envs\*env_name*\Library\bin\tessdata
            # can be downloaded from https://github.com/tesseract-ocr/tessdata/blob/master/deu.traineddata
            text = pytesseract.image_to_string(img, lang='deu')
            date_str = fname[4:12]  # cut out 8 digit date
            date = datetime.datetime.strptime(date_str, '%Y%m%d')
            data.append([date, text, i, fname])

    # save final df
    df = pd.DataFrame(data, columns=columns)
    df.to_excel('data/articles_raw_gen{}.xlsx'.format(datetime.date.today()), engine='xlsxwriter')


    print("Saved text in DataFrame. Elapsed time: {}".format(time.strftime("%Mm %Ss", time.gmtime(time.time()-start))))


# %% Run file
if __name__ == '__main__':
    main()
