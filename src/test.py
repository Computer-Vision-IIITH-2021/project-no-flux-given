import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from IM-AE import IM_AE

import argparse

parser = argparse.ArgumentParser()

parser.add_argument( "--learning_rate", action="store", dest="learning_rate", default=0.00005, type=float,
                     help="Learning rate for adam [0.00005]" )
parser.add_argument( "--beta1", action="store", dest="beta1", default=0.5, type=float,
                     help="Momentum term of adam [0.5]" )
parser.add_argument( "--dataset", action="store", dest="dataset", default="all_vox256_img", help="The name of dataset" )
parser.add_argument( "--checkpoint_dir", action="store", dest="checkpoint_dir", default="checkpoint",
                     help="Directory name to save the checkpoints [checkpoint]" )
parser.add_argument( "--data_dir", action="store", dest="data_dir", default="./data/all_vox256_img/",
                     help="Root directory of dataset [data]" )
parser.add_argument( "--sample_dir", action="store", dest="sample_dir", default="./samples/",
                     help="Directory name to save the image samples [samples]" )
parser.add_argument( "--sample_vox_size", action="store", dest="sample_vox_size", default=64, type=int,
                     help="Voxel resolution for coarse-to-fine training [64]" )
parser.add_argument( "--start", action="store", dest="start", default=0, type=int,
                     help="In testing, output shapes [start:end]" )
parser.add_argument( "--end", action="store", dest="end", default=16, type=int,
                     help="In testing, output shapes [start:end]" )
parser.add_argument("--testz", action="store_true", dest="testz", default=False,
                     help="True for testing latent codes [False]")


FLAGS = parser.parse_args()

if not os.path.exists( FLAGS.sample_dir ):
    os.makedirs( FLAGS.sample_dir )

im_ae = IM_AE(FLAGS, train=False )

if FLAGS.testz:
    im_ae.test_z(FLAGS)
else:
    im_ae.test_mesh_point( FLAGS )
