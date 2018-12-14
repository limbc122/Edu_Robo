import csv
from urllib.request import urlopen, Request
from bs4 import BeautifulSoup as bs

def finviz(ticker):
    f = open(ticker + '.csv', 'w', encoding='utf-8', newline='')
    writer = csv.writer(f)

    url = "https://finviz.com/quote.ashx?t="+ticker
    req = Request(url)
    html = urlopen(req).read()
    soup = bs(html, 'html.parser')

    names = soup.findAll(class_='snapshot-td2-cp')
    datas = soup.findAll(class_='snapshot-td2')

    for i in range(len(names)):
        name = names[i].getText()
        data = datas[i].getText()
        writer.writerow((ticker, name, data))
    
    f.close()

finviz("NVDA")