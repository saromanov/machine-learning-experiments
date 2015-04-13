import urllib.request

def get_german(outpath):
	""" output path for load """
	path = 'https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data-numeric'
	data = urllib.request.urlopen(path)
	with open(outpath, 'wb') as writer:
		writer.write(data.read())
