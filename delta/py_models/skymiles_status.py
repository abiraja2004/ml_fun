import csv
import sys

rownum = 0
ifile = open(sys.argv[1], 'rb')
reader = csv.reader(ifile)
ofile = open(sys.argv[2], 'wb')
writer = csv.writer(ofile, delimiter=',',quotechar='#')
statuses = {}
statuses['n'] = 0
statuses['s'] = 0
statuses['g'] = 0
statuses['p'] = 0
statuses['d'] = 0
for row in reader:
    status = 'n'
    if rownum == 0:
        hdr = row
    else:
        mqm = float(row[31])/1000.
        mqs = float(row[17])
        mqd = float(row[18])/1000.
        if (((mqm >= 25) or (mqs >= 30)) and (mqd >= 3)):
            status = 's'
        if (((mqm >= 50) or (mqs >= 60)) and (mqd >= 6)):
            status = 'g'
        if (((mqm >= 75) or (mqs >= 100)) and (mqd >= 9)):
            status = 'p'
        if (((mqm >= 125) or (mqs >= 140)) and (mqd >= 15)):
            status = 'd'
    statuses[status] += 1
    row.append(status)
    writer.writerow(row)
    rownum += 1
print statuses
ssum = float(sum(statuses.itervalues()))
for k, v in statuses.iteritems():
    print k, round(float(100.*v/ssum),2)
ifile.close()
ofile.close()
