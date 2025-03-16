import torch
import torch.nn.functional as F

def temp_denoise(model, noisyframe, noise_p, noise_g,):
	'''Encapsulates call to denoising model and handles padding.
		Expects noisyframe to be normalized in [0., 1.]
	'''
	# make size a multiple of four (we have two scales in the denoiser)
	sh_im = noisyframe.size()
	expanded_h = sh_im[-2]%32
	if expanded_h:
		expanded_h = 32-expanded_h
	expanded_w = sh_im[-1]%32
	if expanded_w:
		expanded_w = 32-expanded_w
	padexp = (0, expanded_w, 0, expanded_h)
	noisyframe = F.pad(input=noisyframe, pad=padexp, mode='reflect')
	noise_p = F.pad(input=noise_p, pad=padexp, mode='reflect')
	noise_g = F.pad(input=noise_g, pad=padexp, mode='reflect')

	noise_map = torch.cat((noise_p, noise_g), 1) / (2**12-1 - 240)
	# denoise
	out = model(noisyframe, noise_map)

	if expanded_h:
		out = out[:, :, :-expanded_h, :]
	if expanded_w:
		out = out[:, :, :, :-expanded_w]

	return out

def temp_denoise_single(model, noisyframe, noise_p, noise_g,):
	'''Encapsulates call to denoising model and handles padding.
		Expects noisyframe to be normalized in [0., 1.]
	'''
	# make size a multiple of four (we have two scales in the denoiser)
	sh_im = noisyframe.size()
	expanded_h = sh_im[-2]%32
	if expanded_h:
		expanded_h = 32-expanded_h
	expanded_w = sh_im[-1]%32
	if expanded_w:
		expanded_w = 32-expanded_w
	padexp = (0, expanded_w, 0, expanded_h)
	noisyframe = F.pad(input=noisyframe, pad=padexp, mode='reflect')
	noise_p = F.pad(input=noise_p, pad=padexp, mode='reflect')
	noise_g = F.pad(input=noise_g, pad=padexp, mode='reflect')

	noise_map = torch.cat((noise_p, noise_g), 1) / (2**12-1 - 240)
	# denoise
	out = model.module.forward_single_image(noisyframe[:,3:4], noise_map)

	if expanded_h:
		out = out[:, :, :-expanded_h, :]
	if expanded_w:
		out = out[:, :, :, :-expanded_w]

	return out

def denoise_seq(seq, noise_p, noise_g, temp_psz, model_temporal):

	# init arrays to handle contiguous frames and related patches
	numframes, H, W = seq.shape
	noise_p = noise_p.expand((1, 1, H, W))
	noise_g = noise_g.expand((1, 1, H, W))


	ctrlfr_idx = int((temp_psz-1)//2)
	inframes = list()
	denframes = torch.empty((numframes, H, W)).to(seq.device)
	denframes_SFD = torch.empty((numframes, H, W)).to(seq.device)

	# build noise map from noise std---assuming Gaussian noise

	for fridx in range(numframes):
		if not inframes:
			# if list not yet created, fill it with temp_patchsz frames
			for idx in range(temp_psz):
				relidx = abs(idx - ctrlfr_idx)  # handle border conditions, reflect
				inframes.append(seq[relidx])
		else:
			del inframes[0]
			relidx = min(fridx + ctrlfr_idx, -fridx + 2 * (numframes - 1) - ctrlfr_idx)  # handle border conditions
			inframes.append(seq[relidx])

		if numframes - ctrlfr_idx > fridx >= ctrlfr_idx:
			inframes_t = torch.stack(inframes, dim=0).contiguous().view((1, temp_psz, H, W)).to(seq.device)
			# append result to output list
			out = temp_denoise(model_temporal, inframes_t, noise_p, noise_g)
			denframes[fridx] = torch.squeeze(out)
			out = temp_denoise_single(model_temporal, inframes_t, noise_p, noise_g)
			denframes_SFD[fridx] = torch.squeeze(out)

	# free memory up
	del inframes
	del inframes_t
	torch.cuda.empty_cache()

	# convert to appropiate type and return
	return denframes, denframes_SFD
