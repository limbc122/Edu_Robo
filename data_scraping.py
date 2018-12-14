import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from urllib.request import urlopen, Request
from bs4 import BeautifulSoup as bs

def get_html(ticker, gb):
    url = []
    url.append("https://comp.fnguide.com/SVO2/ASP/SVD_Finance.asp?pGB=1&gicode=A" + ticker + "&cID=&MenuYn=Y&ReportGB=&NewMenuID=103&stkGb=701")
    url.append("https://comp.fnguide.com/SVO2/ASP/SVD_FinanceRatio.asp?pGB=1&gicode=A" + ticker + "&cID=&MenuYn=Y&ReportGB=&NewMenuID=104&stkGb=701")
    url.append("https://comp.fnguide.com/SVO2/ASP/SVD_Invest.asp?pGB=1&gicode=A" + ticker + "&cID=&MenuYn=Y&ReportGB=&NewMenuID=105&stkGb=701")
    url.append("https://comp.fnguide.com/SVO2/ASP/SVD_Consensus.asp?pGB=1&gicode=A" + ticker +"&cID=&MenuYn=Y&ReportGB=&NewMenuID=108&stkGb=701")
    url = url[gb]

    req = Request(url, headers={"User-Agent":'Mozila/5.0'})
    html_text = urlopen(req).read()
    return html_text

def ext_fin(ticker, gb, item, n, freq='a'):
    html_text = get_html(ticker, gb)
    soup = bs(html_text, 'html.parser')
    d = soup.findAll(text=item)
    nlimt = 3 if gb == 0 else 4

    if freq == 'a':
        d = d[0].find_all_next(class_='r', limit=nlimt)
    else:
        d = d[1].find_all_next(class_='r', limit=nlimt)

    data = d[(nlimt-n):nlimt]
    v = [v.text for v in data]
    return (v)

ext_fin('005930', 0, '매출액', 3)


# html = get_html('005930', 0)
# html = urlopen('http://www.pythonscraping.com/pages/page1.html')
# soup = bs(html, 'html.parser')
# soup.h1

