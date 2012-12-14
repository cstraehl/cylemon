"""
Usage: preprocess.py [-h] INPUT OUTPUT [-s sigma]  

INPUT       h5 file that contains a 3d or 5d dataset at path volume/data
OUTPUT      the name of the output .graph5 file
-s sigma    scale of the presmoothing for the hessian of gaussian eigenvalues [default: 1.3]
-h --help   show this description
"""
from docopt import docopt
import sys
arguments = docopt(__doc__, argv=sys.argv[1:], help=True, version=None)

import numpy
import time
import h5py
import pyximport; pyximport.install()
from cylemon.segmentation import MSTSegmentor


import vigra
import scipy.ndimage
import numpy

inputf=arguments["INPUT"]
outputf=arguments["OUTPUT"]

print "preprocessing file %s to outputfile %s" % (inputf, outputf)

sigma = float(arguments["-s"])

h5f = h5py.File(inputf,"r")

volume = h5f["volume/data"][:]
if volume.ndim == 5:
    volume = volume[0,:,:,:,0]

print "input volume shape: ", volume.shape
print "input volume size: ", volume.nbytes / 1024**2, "MB"
fvol = volume.astype(numpy.float32)
print "Hessian Eigenvalues..."
volume_feat = vigra.filters.hessianOfGaussianEigenvalues(fvol,sigma)[:,:,:,0]
volume_ma = numpy.max(volume_feat)
volume_mi = numpy.min(volume_feat)
volume_feat = (volume_feat - volume_mi) * 255.0 / (volume_ma-volume_mi)
print "Watershed..."
labelVolume = vigra.analysis.watersheds(volume_feat)[0].astype(numpy.int32)

print labelVolume.shape, labelVolume.dtype
mst = MSTSegmentor(labelVolume, volume_feat.astype(numpy.float32), edgeWeightFunctor = "minimum")
mst.raw = volume

mst.saveH5(outputf,"graph")

