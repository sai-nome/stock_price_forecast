# SBI証券から海外のティッカーまたはコードと銘柄を取得
from bs4 import BeautifulSoup
import re, sys, time
import requests
import pandas as pd

sbi_foreign_stock_list = [
  {"america":   {"url": "https://search.sbisec.co.jp/v2/popwin/info/stock/pop6040_usequity_list.html", "tag": "table", "c_cd":"", "class": "md-l-table-01 md-l-utl-mt10"}},
  {"china":     {"url": "https://search.sbisec.co.jp/v2/popwin/info/stock/pop6040_hk_list.html",       "tag": "div",   "c_cd":"HK", "class": "accTbl01"}},
  {"korea":     {"url": "https://search.sbisec.co.jp/v2/popwin/info/stock/pop6040_kr_list.html",       "tag": "div",   "c_cd":"KS", "class": "accTbl01"}},
  {"russia":    {"url": "https://search.sbisec.co.jp/v2/popwin/info/stock/pop6040_ru_list.html",       "tag": "div",   "c_cd":"ME", "class": "accTbl01"}},
  {"vietnam":   {"url": "https://search.sbisec.co.jp/v2/popwin/info/stock/pop6040_vn_list.html",       "tag": "div",   "c_cd":"VN", "class": "accTbl01"}},
  {"Indonesia": {"url": "https://search.sbisec.co.jp/v2/popwin/info/stock/pop6040_id_list.html",       "tag": "div",   "c_cd":"JK", "class": "accTbl01"}},
  {"Singapore": {"url": "https://search.sbisec.co.jp/v2/popwin/info/stock/pop6040_sg_list.html",       "tag": "div",   "c_cd":"SI", "class": "accTbl01"}}, # SBIとyahooのティカーが一致しない件
  {"thailand":  {"url": "https://search.sbisec.co.jp/v2/popwin/info/stock/pop6040_th_list.html",       "tag": "div",   "c_cd":"BK", "class": "accTbl01"}},
  {"malaysia":  {"url": "https://search.sbisec.co.jp/v2/popwin/info/stock/pop6040_my_list.html",       "tag": "div",   "c_cd":"KL", "class": "accTbl01"}} # SBIとyahooのティカーが一致しない件
]
out_oth_fol = "/stock_list/overseas_list/"

for foreign_dict in sbi_foreign_stock_list:
  for country, values in foreign_dict.items():
    country_url = values["url"]
    country_tag = values["tag"]
    country_code = values["c_cd"]
    country_class = values["class"]
    html = requests.get(country_url)
    soup = BeautifulSoup(html.content, "html.parser")
    for table_soup in soup.find_all(country_tag, class_=country_class):
      stock_data = []
      rows = table_soup.find_all("tr")
      for row in rows[1:]:
        ticker = row.find_all("th")[0].text.strip()
        cols = row.find_all("td")
        name = cols[0].text.strip()
        stock_data.append((ticker, name))
      continue

    df_foreign_stock = pd.DataFrame(stock_data, columns=["コード", "銘柄名"])
    df_foreign_stock["コード"] = df_foreign_stock["コード"].astype(str) + country_code
    df_foreign_stock.to_csv(out_oth_fol + country + "_stock_list.csv")
    time.sleep(2)
