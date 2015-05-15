from urllib.request import urlopen
import tarfile
import os


def get_indoor(outdir):
    name = 'indoorCVPR_09.tar'
    path = 'http://groups.csail.mit.edu/vision/LabelMe/NewImages/indoorCVPR_09.tar'
    result = urlopen(path)
    with open(outdir + '/' + name, 'wb') as writer:
        writer.write(result.read())
    result = tarfile.openfile(outdir + '/' + name)
    result.extract_all(path=outfir + '/indoorCVPR09')
    result.close()



def _get_training_and_test_names(outdir):
    trainname = 'TrainImages.txt'
    testname = 'TestImages.txt'
    trainimages = 'http://web.mit.edu/torralba/www/TrainImages.txt'
    testimages = 'http://web.mit.edu/torralba/www/TestImages.txt'
    def get_data(path, name):
        result = urlopen(path)
        with open(outdir + '/' + name, 'wb') as writer:
            writer.write(result.read())
    if not os.path.exists(outdir + '/' + trainname):
        get_data(trainimages, trainname)
    if not os.path.exists(outdir + '/' + testname):
        get_data(testimages, testname)
