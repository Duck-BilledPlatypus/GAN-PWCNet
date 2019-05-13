import PWCNet
import sys
import cv2
import torch
import numpy as np
from math import ceil
from torch.autograd import Variable
from scipy.ndimage import imread
import flowlib
"""
Contact: Deqing Sun (deqings@nvidia.com); Zhile Ren (jrenzhile@gmail.com)
"""


def calvalidation(flo, gt):

	true_flow = gt
	true_flow_sum = true_flow[:, :, 0] + true_flow[:, :, 1]
	valid_index         = np.argwhere(true_flow_sum != 0.0)
	invalid_index       = np.argwhere(true_flow_sum == 0.0)
	pred_diff     = (flo[:, :, 0] - gt[:, :, 0])**2 + \
					(flo[:, :, 1] - gt[:, :, 1])**2
	pred_diff     = pred_diff**0.5
	for i in range(len(invalid_index)):
		pred_diff[invalid_index[i][0], invalid_index[i][1]] = 0.0

	incorrect_count = 0
	for i in range(len(valid_index)):
		incorrect_count += (pred_diff[valid_index[i][0], valid_index[i][1]] >= 3.0)

	validation_loss = float(incorrect_count)/float(len(valid_index))
	return validation_loss

def writeFlowFile(filename,uv):
# 	"""
# 	According to the matlab code of Deqing Sun and c++ source code of Daniel Scharstein  
# 	Contact: dqsun@cs.brown.edu
# 	Contact: schar@middlebury.edu
# 	"""
	TAG_STRING = np.array(202021.25, dtype=np.float32)
	if uv.shape[2] != 2:
		sys.exit("writeFlowFile: flow must have two bands!");
	H = np.array(uv.shape[0], dtype=np.int32)
	W = np.array(uv.shape[1], dtype=np.int32)
	with open(filename, 'wb') as f:
		f.write(TAG_STRING.tobytes())
		f.write(W.tobytes())
		f.write(H.tobytes())
		f.write(uv.tobytes())
		
def child(im1_fn, im2_fn, flow_fn, pwc_model_fn, gt_fn):
		# if len(sys.argv) > 1:
		#     im1_fn = sys.argv[1]
		# if len(sys.argv) > 2:
		#     im2_fn = sys.argv[2]
		# if len(sys.argv) > 3:
		#     flow_fn = sys.argv[3]



		im_all = [imread(img) for img in [im1_fn, im2_fn]]
		im_all = [im[:, :, :3] for im in im_all]

		# rescale the image size to be multiples of 64
		divisor = 64.
		H = im_all[0].shape[0]
		W = im_all[0].shape[1]

		H_ = int(ceil(H/divisor) * divisor)
		W_ = int(ceil(W/divisor) * divisor)
		for i in range(len(im_all)):
			im_all[i] = cv2.resize(im_all[i], (W_, H_))

		for _i, _inputs in enumerate(im_all):
			im_all[_i] = im_all[_i][:, :, ::-1]
			im_all[_i] = 1.0 * im_all[_i]/255.0
			
			im_all[_i] = np.transpose(im_all[_i], (2, 0, 1))
			im_all[_i] = torch.from_numpy(im_all[_i])
			im_all[_i] = im_all[_i].expand(1, im_all[_i].size()[0], im_all[_i].size()[1], im_all[_i].size()[2])	
			im_all[_i] = im_all[_i].float()
		    
		im_all = torch.autograd.Variable(torch.cat(im_all,1).cuda(), volatile=True)

		print('start')
		net = PWCNet.pwc_dc_net(pwc_model_fn)
		net = net.cuda()
		net.eval()

		flo = net(im_all)
		flo = flo[0] * 20.0
		flo = flo.cpu().data.numpy()

		# scale the flow back to the input size 
		flo = np.swapaxes(np.swapaxes(flo, 0, 1), 1, 2) # 
		u_ = cv2.resize(flo[:,:,0],(W,H))
		v_ = cv2.resize(flo[:,:,1],(W,H))
		u_ *= W/ float(W_)
		v_ *= H/ float(H_)
		flo = np.dstack((u_,v_))

		gt = flowlib.read_flow(gt_fn)


		valiloss = calvalidation(flo, gt)

		writeFlowFile(flow_fn, flo)

		return valiloss
def main():
	img1path = '/home/lyc/Desktop/Oxford_6000/training/20141210/000'
	img2path = '/home/lyc/Desktop/Oxford_6000/training/20141210/000'
	gtpath = '/home/lyc/Desktop/Oxford_6000/flow/20141210/'
	pthpath = '/home/lyc/Desktop/Synthetic2Realistic_original/checkpoints/tpovr/'

	pwc_model_fn = ['pwc_net.pth.tar'] + [str(i + 1) + '_net_img2task.pth' for i in range(39,40)]
	print(pwc_model_fn)
	loss = [0] * len(pwc_model_fn)
	for i in range(len(pwc_model_fn)):
		print('i', i)
		pth = pwc_model_fn[i]
		for j in range(5):
			print('j', j)
			loss[i] += child(img1path+str(j+1)+'.png',img2path+str(j+2)+'.png',str(i + 1)+str(j + 1)+'.flo',pthpath + pth,gtpath+str(j+1)+'.flo')
		loss[i] /= 5
	print(loss)


	

if __name__ == '__main__':
	main()

