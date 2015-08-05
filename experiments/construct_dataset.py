from fuel.datasets.hdf5 import H5PYDataset
import h5py
import numpy

numpy.save('train_vec.npy', numpy.random.normal(size=(90,10)).astype('float32'))
numpy.save('target_vec.npy', numpy.random.normal(size=(90,1)).astype('uint8'))

train_vec = numpy.load('train_vec.npy')
target_vec = numpy.load('target_vec.npy')
f = h5py.File('dataset1.hdf5', mode='w')
vector = f.create_dataset('train_vec', (90,10), dtype='float32')
targets = f.create_dataset('target_vec', (90,1), dtype='uint8')
vector[...] = numpy.vstack([train_vec])
targets[...] = numpy.vstack([target_vec])
vector.dims[0].label = 'batch'
vector.dims[1].label = 'feature'
targets.dims[0].label = 'batch'
targets.dims[1].label = 'index'
split_dict = {
    'train': {'train_vec': (0,90), 'target_vec': (0,90)}
}
f.attrs['split'] = H5PYDataset.create_split_array(split_dict)
f.flush()
f.close()

load_train = H5PYDataset('dataset1.hdf5', which_sets=('train',), subset=slice(0,80))
print(load_train.open())