import threading
import requests
import time

start = time.time()

def fetch_url(url):
    r = requests.get(url, allow_redirects=False)
    if r.status_code == 200:
        cnt += 1

urls = []
append = urls.append
for url in open('test_urls.dat'):
    append(url)

cnt = 0
threads = [threading.Thread(target=fetch_url, args=(url,)) for url in urls]
for thread in threads:
    thread.start()
for thread in threads:
    thread.join()

print "Elapsed Time: %s" % (time.time() - start)
                                
