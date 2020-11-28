import scipy.io as scio

file = '/media/data/DataSet/DETRAC_TRACK/DETRAC-Test-Annotations-MAT/DETRAC-Test-Annotations-MAT/MVI_39031.mat'

data = scio.loadmat(file)
print(data)