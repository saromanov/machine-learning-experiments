import urllib.request
import gzip
import os

def get_github(year=2015, month=4, day=26, hour=21):
	path = 'github-{0}-{1}-{2}-{3}.json'.format(year, month, day, hour)
	if os.path.exists(path):
		return 
	pathgz = path + '.gz'
	pre = 'http://data.githubarchive.org/{0}-{1}-{2}-{3}.json.gz'
	if month < 10:
		pre = 'http://data.githubarchive.org/{0}-0{1}-{2}-{3}.json.gz'
	url = pre.format(year, month, day, hour)
	data = urllib.request.urlopen(url)
	fs = open(pathgz, 'wb')
	fs.write(data.read())
	fs.close()

	#extract
	with gzip.open(pathgz, 'rb') as gzfile:
		with open(path, 'wb') as outfile:
			for line in gzfile:
				outfile.write(line)
