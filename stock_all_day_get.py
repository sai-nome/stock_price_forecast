inp_jap_fol = "./stock_list/japan_list/"
inp_oth_fol = "./stock_list/overseas_list/"
out_fol = "./stock_data/"

import pandas as pd
import glob
inp_file_jap_list = glob.glob(inp_jap_fol + "*.csv")
inp_file_oth_list = glob.glob(inp_oth_fol + "*.csv")

# from pandas_datareader.yahoo.daily import YahooDailyReader
# import pandas_datareader
import pandas_datareader.data as web
import yfinance as yf
import datetime, time

# 取得したい期間を設定
# 株価が出始めた時点から現在までのデータを取得
# 開始日を非常に古い日付（例: 1900年1月1日）に設定し、終了日には現在の日付を使用
start = datetime.datetime(1900, 1, 1)
end = datetime.datetime.now()

# Yahoo Financeからデータを取得
yf.pdr_override() #追加
for jap_file_list in inp_file_oth_list:
  jap_list = pd.read_csv(jap_file_list).values.tolist()
  for stock in jap_list:
    # df = yf.download(str(stock[1]), start, end)
    # 東京はT、札幌はSが後ろにつくので注意、アメリカは不要
    # df = web.get_data_yahoo([str(stock[1])+".T"], start, end)
    df = web.get_data_yahoo([str(stock[1])], start, end)
    print(df.info)
    df.to_csv(out_fol+stock[2].replace("/", " ").replace("\r\n", " ").replace("?", " ")+".csv")
    time.sleep(2)
# 取得したデータを表示
# df
