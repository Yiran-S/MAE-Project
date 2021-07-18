#!/usr/bin/env python
# coding: utf-8

# In[1]:


import bs4
from bs4 import BeautifulSoup
import requests
import pandas as pd
import re
from datetime import datetime

import pymysql


# In[2]:


page = requests.get("http://www.estesparkweather.net/archive_reports.php?date=202104")
soup = BeautifulSoup(page.content,"html.parser")

#print(soup.prettify())

table = soup.find_all("table")
raw_data = [row.text.splitlines() for row in table]
raw_data = raw_data[:-9]
for i in range(len(raw_data)):
    raw_data[i] = raw_data[i][2:len(raw_data[i]):3]

#clean data
df_list= list()
ave, rain, temp, humid = list(),list(),list(),list()
pressure, wind, gust = list(),list(),list()

for i in range(len(raw_data)):
    c = ['.'.join(re.findall("\d+",str(raw_data[i][j].split()[:5])
                            ))for j in range(len(raw_data[i]))]
    df_list.append(c)
    df_list[i][0] = '202104'+ df_list[i][0]
    df_list[i][0] = datetime.strptime(str(df_list[i][0]), 
                                      '%Y%m%d').strftime('%Y-%m-%d')
    ave.append(df_list[i][0:8])
    rain.append([df_list[i][0],df_list[i][8],df_list[i][9],df_list[i][10]])
    temp.append([df_list[i][0],df_list[i][1],df_list[i][11],
                 df_list[i][12],df_list[i][-1]])
    humid.append([df_list[i][0],df_list[i][3],df_list[i][13],df_list[i][14]])
    pressure.append([df_list[i][0],df_list[i][4],df_list[i][15],df_list[i][16]])
    wind.append([df_list[i][0],df_list[i][5],df_list[i][17]])
    gust.append([df_list[i][0],df_list[i][6],df_list[i][18]])
    


# In[4]:


endpoint = "weather.c7d2tmdwfvx6.us-east-2.rds.amazonaws.com"
dbname = "AWSweather"
username = 'admin'
password = 'Iloveyou'

connection = pymysql.connections.Connection(host = endpoint, user=username, 
                                            password = password,database = dbname)


cursor = connection.cursor()
query1 = "INSERT INTO Ave_weather VALUES (%s, %s, %s, %s, %s, %s, %s, %s)" 
cursor.executemany(query1, ave)
connection.commit()

query2 = "INSERT INTO rain VALUES (%s, %s, %s, %s)"
cursor.executemany(query2, rain)
connection.commit()

query3 = "INSERT INTO temperature VALUES (%s, %s, %s, %s, %s)"
cursor.executemany(query3, temp)
connection.commit()

query4 = "INSERT INTO humidity VALUES (%s, %s, %s, %s)"
cursor.executemany(query4, humid)
connection.commit()

query5 = "INSERT INTO pressure VALUES (%s, %s, %s, %s)"
cursor.executemany(query5, pressure)
connection.commit()

query6 = "INSERT INTO wind VALUES (%s, %s, %s)"
cursor.executemany(query6, wind)
connection.commit()

query7 = "INSERT INTO gust VALUES (%s, %s, %s)"
cursor.executemany(query7, gust)
connection.commit()

#flattened_values = [item for sublist in values_to_insert for item in sublist]
#c.execute(query, flattened_values)

