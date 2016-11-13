import os
import glob

print 'Starting Test Set'

for name in glob.glob('/ASTRO_Research/Test_Images/*'):
    print '\n\n\n'
    print name[29:]
    print '------------------------'
    os.system('python /ASTRO_Research/label_image.py ' + name)
    print name[29:]
    print '------------------------'
