import requests
from bs4 import BeautifulSoup
from urlparse import urlparse
from threading import Thread
import httplib
import sys
from Queue import Queue


def doWork():
    while True:
        url = q.get()
        status, url = getStatus(url)
        if status == 200:
            open_url(url)
        q.task_done()

        
def getStatus(ourl):
    try:
        url = urlparse(ourl)
        conn = httplib.HTTPConnection(url.netloc)
        conn.request("HEAD", url.path)
        res = conn.getresponse()
        return res.status, ourl
    except:
        return "error", ourl

    
def open_url(url):
    r = requests.get(url, allow_redirects=False)
    soup = BeautifulSoup(r.text, 'html.parser')

    # player name
    player = {}
    try:
        a = soup.find('div', {'class': 'player-name'})
        #player['espn_id'] = pid
        player['name'] = a.find('h1').contents[0]
    except:
        return
    
    # player metrics
    a = soup.find('div', id='tab4')
    try:
        b = a.findAll('li', {'class': 'combine-list'})
        for c in b:
            k = c.find('div', {'class': 'combine-id'}).contents[0]
            v = c.find('li', {'class': 'combine-bar-data'}).contents[0]
            player[str(k)] = v
        print player
        print url
        ofile.write(url+'\n')
    except:
        return


pstart = 22000
pend = 220000
ids = range(pstart, pend)
ofile = open('test_urls.dat','w')
for pid in ids:
    ofile.write('http://espn.go.com/college-sports/football/recruiting/player/combine/_/id/'+str(pid)+'\n')
ofile.close()

concurrent = 200
q = Queue(concurrent * 2)
ofile = open('good_urls.dat', 'w')
for i in range(concurrent):
    t = Thread(target=doWork)
    t.daemon = True
    t.start()
try:
    for url in open('test_urls.dat'):
        q.put(url.strip())
    q.join()
except KeyboardInterrupt:
    sys.exit(1)
ofile.close()

