# Learning Implicit Fields for Generative Shape Modelling
#### Please find the instructions for running the code below

### Installation Details
```bash
pip install numpy
pip install scipy
pip install h5py
pip install pytorch
pip install PyMCubes

```

The link to the dataset after preprocessing is:
https://drive.google.com/file/d/1yh5EitNemKXaurRYX7YdHUWAbfHtEF12/view?usp=sharing

### Training IM-AE:
Progressive training 
```bash
python3 train.py --epoch 200 --sample_vox_size 16
python3 train.py --epoch 200 --sample_vox_size 32
python3 train.py --epoch 200 --sample_vox_size 64
```

### Testing IM-AE:
```bash
python3 test.py --sample_dir samples/im_ae_out --start 0 --end 8
```

### Training IMGAN:
```bash
python3 train.py --getz
```

### Testing IMGAN:
```bash
python3 test.py --testz --sample_dir samples/gan_out --start 0 --end 8
```

