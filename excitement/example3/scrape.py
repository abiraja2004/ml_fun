#from BeautifulSoup import BeautifulSoup
import bs4
import urllib2
import pandas as pd
import numpy as np
import requests
import dateutil
import QSTK
#import copper

#url="http://www.utexas.edu/world/univ/alpha/"
url="http://espn.go.com/nba/teams"
#page=urllib2.urlopen(url)
#soup = bs4.BeautifulSoup(page.read())
#print soup.prettify()
r = requests.get(url)
soup = bs4.BeautifulSoup(r.text)
tables = soup.find_all('ul', class_='medium-logos')

teams = []
prefix_1 = []
prefix_2 = []
teams_urls = []
for table in tables:
    lis = table.find_all('li')
    for li in lis:
        info = li.h5.a
        #teams begin with h5, as seen below
        #<h5>
         # <a class="bi" href="http://espn.go.com/nba/team/_/name/bos/boston-celtics">
        # Boston Celtics </a>
        #so "Boston Celtics" is in h5.a
        teams.append(info.text)
        url = info['href']
        teams_urls.append(url)
        prefix_1.append(url.split('/')[-2])
        prefix_2.append(url.split('/')[-1])

dic = {'url': teams_urls, 'prefix_2': prefix_2, 'prefix_1': prefix_1}
teams = pd.DataFrame(dic, index=teams)
teams.index.name = 'team'
print(teams)
#copper.save(teams, 'teams')


#print soup.prettify()


#universities=soup.findAll('a',{'class':'institution'})
#findAll('a') finds all the `a` tags such as <a href=...
#for eachuniversity in universities:
#    print eachuniversity['href']+","+eachuniversity.string