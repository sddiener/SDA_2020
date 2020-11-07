# %% Setup
import os
import pandas as pd
import time
import datetime

import pdfplumber
import pytesseract
from pdf2image import convert_from_path
from langdetect import detect

# define directories
data_dir = "data/"

dly_new_dir = data_dir + "SNB_Tagesspiegel_new/"
dly_old_dir = data_dir + "SNB_Tagesspiegel_old/"
wk_new_dir = data_dir + "SNB_Wochenspiegel_new/"
wk_old_dir = data_dir + "SNB_Wochenspiegel_old/"


# %% Define PDF and Page Class
class PdfDoc:
    pages = None
    imgs = None
    page_format = None
    filename = None

    def __init__(self, path):
        self.pages = pdfplumber.open(path).pages
        self.imgs = convert_from_path(path)
        self.page_format = self.get_page_format()
        self.filename = os.path.basename(path)

    def get_page_format(self):
        """ Determines PDF page_format based on title page. """
        title_page = PdfPage(self.pages[0], self.imgs[0], page_format='all_page', extract_info=False)
        tp_text = title_page.extract_text(page_format='all_page')

        if "Wochenspiegel" in tp_text['body']:
            pdf_format = 'wk_new'
        elif "medienspiegel" in tp_text['body'].lower():
            pdf_format = 'dly_new'
        else:
            pdf_format = 'wk_old'

        return pdf_format


class PdfPage:
    page = None  # pdf plumber obj
    img = None  # page as image
    page_formats = ['wk_new', 'wk_old', 'all_page']
    page_format = None
    isarticle = None
    nr = None
    text = None
    lang = None

    def __init__(self, page, img, page_format, extract_info=False):
        self.page = page
        self.img = img
        self.page_format = page_format
        self.nr = page.page_number
        if extract_info:  # put all self. args here:
            self.lang = self.get_lang()
            self.text = self.extract_text(page_format)
            self.isarticle = self.is_article()

    def get_lang(self):
        bboxes = self.get_bboxes(page_format=self.page_format, convert=False)
        Page_crop = self.page.crop(bboxes['body'])
        text = Page_crop.extract_text()

        lang_dict = {'en': 'eng',
                     'de': 'deu',
                     'fr': 'fra',
                     'it': 'ita'}

        try:
            lang = detect(text)
            lang = lang_dict.get(lang, 'deu+fra+ita+eng')  # if not convertible: tesseract chooses
        except:
            lang = 'deu+fra+ita+eng'

        return lang

    def extract_text(self, page_format):
        """ Extracts text from pdf article based on bboxes. """
        bboxes = self.get_bboxes(page_format=page_format, convert=True)

        text = {}
        for key, bbox in bboxes.items():
            img_crop = self.img.crop(bbox)
            lang = 'deu' if key == 'info' else self.lang
            text[key] = pytesseract.image_to_string(img_crop, lang=lang)

        return text

    def get_bboxes(self, page_format, convert=False):
        """ Constructs bounding box of text in pdf article. """
        width, height = float(self.page.width), float(self.page.height)

        if page_format in ('wk_new', 'dly_new'):  # (w0, h0) is top-left
            w0 = 0
            w1 = width

            h0 = 0
            h1 = height * 0.182
            h2 = height * 0.93

            bboxes = {
                'info': (w0, h0, w1, h1),
                'body': (w0, h1, w1, h2)
            }

        elif page_format == 'wk_old':
            w0 = 0
            w1 = width

            h0 = 0
            h1 = height

            bboxes = {
                # 'info': (0,0,0,0),
                'body': (w0, h0, w1, h1)
            }

        elif page_format == 'all_page':
            w0 = 0
            w1 = width

            h0 = 0
            h1 = height

            bboxes = {
                # 'info': (0,0,0,0),
                'body': (w0, h0, w1, h1)
            }
        else:  # use entire page
            raise ValueError("Unknown page format. Expected: {} got {}".format(self.page_formats, page_format))

        if convert:
            def convert_bbox(bbox, input_pagesize, output_pagesize):
                w0, h0, w1, h1 = bbox
                w_in, h_in = input_pagesize
                w_out, h_out = output_pagesize

                w0 = w0 / w_in * w_out
                w1 = w1 / w_in * w_out

                h0 = h0 / h_in * h_out
                h1 = h1 / h_in * h_out

                bbox = w0, h0, w1, h1
                return bbox

            pdf_size = (width, height)
            img_size = self.img.size
            for key, bbox in bboxes.items():
                bboxes[key] = convert_bbox(bbox, pdf_size, img_size)

        for key, bbox in bboxes.items():
            bboxes[key] = tuple(map(int, bbox))  # convert to int

        return bboxes

    def is_article(self):
        """ Checks if page is an actual article. """
        # new format: contents dont have "Datum:" at top + no contents after page 5
        if self.page_format in ('wk_new', 'dly_new') and self.nr < 5:
            if self.text is None:
                self.text = self.extract_text(page_format=self.page_format)

            if "Datum:" not in self.text['info']:
                is_article = False
            else:
                is_article = True
        # scripts format: never has contents. + No contents after page 5
        elif self.page_format == 'wk_old':
            is_article = True
        else:
            is_article = True

        return is_article


# Miscellaneous Functions
def sec_to_hours(seconds):
    a = int(seconds // 3600)
    b = int((seconds % 3600) // 60)
    c = int((seconds % 3600) % 60)
    d = "{} h {} min {} s".format(a, b, c)
    return d


# %% Main Structure
def main():
    directories = [dly_new_dir, wk_new_dir]

    # i) specify cols for final df
    columns = ['info', 'body', 'nr', 'lang', 'isarticle', 'page_format', 'filename']
    data = []

    start_time0 = time.time()

    # ii) loop through data directories
    for pdf_dir in directories:
        pdf_files = os.listdir(pdf_dir)

        # iii) loop through pdf files
        for i, pdf_file in enumerate(pdf_files):
            start_time1 = time.time()

            # iv) initialize PDF Document Class
            path = pdf_dir + pdf_file
            PDF = PdfDoc(path)

            # v) loop through pages in PDF object
            for page, img in zip(PDF.pages, PDF.imgs):
                # vi) initialize dedicated Page Class
                Page = PdfPage(page, img, PDF.page_format, extract_info=True)

                # vii) extract info + append to data list
                info = Page.text.get('info', '')
                body = Page.text.get('body', '')

                data.append([info, body, Page.nr, Page.lang, Page.isarticle, Page.page_format, PDF.filename])

            runtime = sec_to_hours(time.time() - start_time1)
            print("Loaded \"{}\" ({}/{}). Elapsed time: {}".format(pdf_file, i + 1, len(pdf_files), runtime))

    # final df
    df = pd.DataFrame(data, columns=columns)

    # save
    df.to_pickle(data_dir + 'articles_raw_gen{}.pkl'.format(datetime.date.today()))

    runtime = sec_to_hours(time.time() - start_time0)
    print("Finished! {} PDFs saved as DataFrame. Elapsed time: {}".format(len(pdf_files), runtime))


if __name__ == '__main__':
    main()
