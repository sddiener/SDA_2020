from bs4 import BeautifulSoup
import urllib.request
import requests
import os


#parse the SNB website
parser = 'html.parser'  # or 'lxml' (preferred) or 'html5lib', if installed
main_url = 'https://www.snb.ch'
resp = urllib.request.urlopen(main_url+'/en/iabout/monpol/id/monpol_current#')
soup = BeautifulSoup(resp, parser, from_encoding=resp.info().get_param('charset'))

#extract all links
links = []
for link in soup.find_all('a', href=True):
    links.append(link['href'])
    
#only keep links with /en/mmr/reference/pre_
links_lagebeurteilung = [main_url+s for s in links if "/en/mmr/reference/pre_" in s]

#specify names of files
names=[]
for i in links_lagebeurteilung:
    test = str(i.split('/')[-1]) #only take the last part of url for the name
    test = str(test.split('.')[-3]) #remove the points
    test = test+'.pdf' #add pdf ending to name
    names.append(test)

#specifiy download path
os.chdir('/Users/jan_s/Desktop/test')
    
# download the file to specific path
def downloadFile(url, fileName):
    with open(fileName, "wb") as file:
        response = requests.get(url)
        file.write(response.content)

for idx, url in enumerate(links_lagebeurteilung):
    downloadFile(url, names[idx])
    
# -----------------------------------------------------------------------------
# Testing the read in of files
#import pdfplumber
#with pdfplumber.open(names[idx]) as pdf:
#    first_page = pdf.pages[0]
#    print(first_page.extract_text())
#    text = first_page.extract_text()

