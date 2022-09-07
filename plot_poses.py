import numpy as np
import pickle

class Isometry():
	"""docstring for Isometry"""
	def __init__(self, R, t):
		self.R = R
		self.t = t
		self.cam = np.eye(4)
		self.cam[:3,:3] = R
		self.cam[:3,3] = t

	def mat(self,):
		return self.cam
		

with open('train_02_scene_info.pkl', "rb") as fp:
  scene = pickle.load(fp)

for k,v in scene.items():
	print('k,v ',k,type(v))

print('scene ',scene['pose'])

# Get camera poses. I am assuming you have some standard Rigid transformation
# class like `Isometry(R=.., t=...)`.
for frame in scene['frames']:
  scene_from_frame = Isometry(**frame['pose']).mat()
  for camera in scene['cameras'].values():
    # Get camera pose w.r.t scene (camera to scene transformation)
    camera_from_frame = Isometry(**camera['extrinsics'])
    scene_from_camera = scene_from_frame * camera_from_frame.inverse()
    print('scene_from_camera ',scene_from_camera)