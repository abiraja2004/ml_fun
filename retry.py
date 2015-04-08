import json
import subprocess
from time import sleep
MAX_TRIES = 100
TSLEEP = 60

jsonStr = '{"script":"make third_party"}'
j = json.loads(jsonStr)
shell_script = j['script']

for i in xrange(MAX_TRIES):
    print "Try %i of %i" % (i,MAX_TRIES)
    proc = subprocess.Popen(shell_script, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    (stdout, stderr) = proc.communicate()
    if stderr:
       print "Shell script gave an error:"
       print stderr
       sleep(TSLEEP) # delay for 1 s
    else:
       print stdout
       print "end" # Shell script ran fine.
       break
