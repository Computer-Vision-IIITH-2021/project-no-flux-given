import numpy as np
import h5py
import cv2
import mcubes

data_dict = h5py.File('02691156_vox.hdf5', 'r')
i = 0
shape = 'airplane'
vox = data_dict['voxels'][:]

batch_vox = vox[i:i+1]
batch_vox = np.reshape(batch_vox,[64,64,64])
img1 = np.clip(np.amax(batch_vox, axis=0)*256, 0,255).astype(np.uint8)
cv2.imwrite(shape + '_' + str(i)+"_vox_1.png",img1)
img2 = np.clip(np.amax(batch_vox, axis=1)*256, 0,255).astype(np.uint8)
cv2.imwrite(shape + '_' + str(i)+"_vox_2.png",img2)
img3 = np.clip(np.amax(batch_vox, axis=2)*256, 0,255).astype(np.uint8)
cv2.imwrite(shape + '_' + str(i)+"_vox_3.png",img3)
vertices, triangles = mcubes.marching_cubes(batch_vox, 0.5)
mcubes.export_mesh(vertices, triangles, shape + '_' + str(i)+"_vox.dae", str(i))


points16 = data_dict['points_16'][:]
data_values16 = data_dict['values_16'][:]

batch_points = points16[i,:]
batch_values = data_values16[i,:]
real_model = np.zeros([16,16,16],np.uint8)
real_model[batch_points[:,0],batch_points[:,1],batch_points[:,2]] = np.reshape(batch_values, [-1])
img1 = np.clip(np.amax(real_model, axis=0)*256, 0,255).astype(np.uint8)
cv2.imwrite(shape + '_' + str(i)+"_16_1.png",img1)
img2 = np.clip(np.amax(real_model, axis=1)*256, 0,255).astype(np.uint8)
cv2.imwrite(shape + '_' + str(i)+ "_16_2.png",img2)
img3 = np.clip(np.amax(real_model, axis=2)*256, 0,255).astype(np.uint8)
cv2.imwrite(shape + '_' + str(i)+"_16_3.png",img3)
vertices, triangles = mcubes.marching_cubes(batch_vox, 0.5)
mcubes.export_mesh(vertices, triangles, shape + '_' + str(i)+"_16.dae", str(i))

points32 = data_dict['points_32'][:]
data_values32 = data_dict['values_32'][:]

batch_points = points32[i,:]
batch_values = data_values32[i,:]
real_model = np.zeros([32,32,32],np.uint8)
real_model[batch_points[:,0],batch_points[:,1],batch_points[:,2]] = np.reshape(batch_values, [-1])
img1 = np.clip(np.amax(real_model, axis=0)*256, 0,255).astype(np.uint8)
cv2.imwrite(shape + '_' + str(i)+"_32_1.png",img1)
img2 = np.clip(np.amax(real_model, axis=1)*256, 0,255).astype(np.uint8)
cv2.imwrite(shape + '_' + str(i)+"_32_2.png",img2)
img3 = np.clip(np.amax(real_model, axis=2)*256, 0,255).astype(np.uint8)
cv2.imwrite(shape + '_' + str(i)+"_32_3.png",img3)
vertices, triangles = mcubes.marching_cubes(batch_vox, 0.5)
mcubes.export_mesh(vertices, triangles, shape + '_' + str(i)+"_32.dae", str(i))

points64 = data_dict['points_64'][:]
data_values64 = data_dict['values_64'][:]

batch_points = points64[i,:]
batch_values = data_values64[i,:]
real_model = np.zeros([64,64,64],np.uint8)
real_model[batch_points[:,0],batch_points[:,1],batch_points[:,2]] = np.reshape(batch_values, [-1])
img1 = np.clip(np.amax(real_model, axis=0)*256, 0,255).astype(np.uint8)
cv2.imwrite(shape + '_' + str(i)+"_64_1.png",img1)
img2 = np.clip(np.amax(real_model, axis=1)*256, 0,255).astype(np.uint8)
cv2.imwrite(shape + '_' + str(i)+"_64_2.png",img2)
img3 = np.clip(np.amax(real_model, axis=2)*256, 0,255).astype(np.uint8)
cv2.imwrite(shape + '_' + str(i)+"_64_3.png",img3)
vertices, triangles = mcubes.marching_cubes(batch_vox, 0.5)
mcubes.export_mesh(vertices, triangles, shape + '_' + str(i)+"_64.dae", str(i))
