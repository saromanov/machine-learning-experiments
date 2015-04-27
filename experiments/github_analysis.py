import os
from getdatasets.get_github_data import get_github
import json

get_github()
fs = open('github-2015-4-26-21.json')
data = json.loads(fs.read())
fs.close()

