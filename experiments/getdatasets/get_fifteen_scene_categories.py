import urllib.request
import zipfile
import os


def get_fifteen_scene_categories(outdir):
	""" Output dir after extraction """
	name = 'scene_categories.zip'
	path = 'http://www-cvr.ai.uiuc.edu/ponce_grp/data/scene_categories/' + name
	if not os.path.exists(name):
		data = urllib.request.urlopen(path)
		with open(name, 'wb') as writer:
			writer.write(data.read())
	if not os.path.exists(name):
		raise Exception("File {0} was not downloaded complete".format(name))
	scene = zipfile.ZipFile(name)
	for name in scene.namelist():
		(dirname, filename) = os.path.split(name)
		fullpath = outdir + '/' + dirname
		if not os.path.exists(fullpath):
			os.makedirs(fullpath)
		if filename != '':
			#scene.extract(filename, dirname)
			if not os.path.exists(fullpath + '/' + filename):
				with open(fullpath + '/' + filename, 'wb') as writer:
					writer.write(scene.read(name))
	scene.close()
