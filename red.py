import pandas as pd
import requests
from bs4 import BeautifulSoup


result = requests.get('https://en.wikipedia.org/wiki/List_of_current_members_of_the_United_States_House_of_Representatives').text
# print(pd.read_html(result.content)[0].head(20))

from bs4 import BeautifulSoup
soup = BeautifulSoup(result,'lxml')
# print(soup.prettify())



My_table = soup.find('table',{'id':'votingmembers'})
# print(My_table)
print(pd.read_html(My_table.prettify())[0].head(20))



df = pd.read_html(My_table.prettify())[0]
df['Party']
print(df['Party.1'].head(20))

print(df.columns)



