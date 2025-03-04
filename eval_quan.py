import glob
import os
import lpips
import argparse
from torchmetrics import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from eval_quan_utils import *
from models import CGNet_D2

parser = argparse.ArgumentParser(description="Quantitative evaluation")
parser.add_argument("--data_dir", type=str, default="/mnt/4TB/datasets/DAVIS2Share/val", help='path of data dir')
parser.add_argument('--models', nargs='+', help='paths for model checkpoints', default=[
    '/mnt/4TB/MA_results/L_CGNet_CA_withCascade_1/ckpt_e79.pth',
    '/mnt/4TB/MA_results/L_CGNet_CA_withCascade_FIP_1/ckpt_e79.pth',
])
argspar = vars(parser.parse_args())

PSNR = PeakSignalNoiseRatio(data_range=1.0).cuda()
SSIM = StructuralSimilarityIndexMeasure(data_range=1.0).cuda()
LPIPS = lpips.LPIPS(net='alex').cuda()

GT_path = argspar['data_dir']
maximum_per_sequence = 100

sequences = sorted(os.listdir(GT_path))
sequences_gt = sequences + ['avg']

model_paths = argspar['models']

noise_levels = [10, 20, 30, 40, 50]

results = {
    model: {
        noise_level: {
            name: {
                'WE': Metric('Warping Error', 2, 2),
                'PSNR': Metric('PSNR', 2, 2),
                'SFD': Metric('SFD', 2, 2),
                "SSIM": Metric('SSIM', 2, 2),
                "LPIPS": Metric('LPIPS', 2, 2)
            }
            for name in sequences
        }
        for noise_level in noise_levels
    }
    for model in model_paths
}

for model in model_paths:
    for noise_level in noise_levels:
        results[model][noise_level]['avg'] = {
            'WE': Metric('Warping Error', 0, 0),
            'PSNR': Metric('PSNR', 0, 0),
            'SFD': Metric('SFD', 0, 0),
            "SSIM": Metric('SSIM', 0, 0),
            "LPIPS": Metric('LPIPS', 0, 0)
        }

previous_denoised_dict = {
    model_path: {
        noise_level: None for noise_level in noise_levels
    }
    for model_path in model_paths}

models = []
for model_path in model_paths:
    model_temp = CGNet_D2(num_input_frames=5)
    # Load saved weights
    state_temp_dict = torch.load(model_path, map_location='cuda')
    model_temp = nn.DataParallel(model_temp, device_ids=[0]).cuda()
    state_dict = state_temp_dict['state_dict']
    model_temp.load_state_dict(state_dict)
    models.append(model_temp.eval())


for seq in sorted(sequences):

    files = sorted(glob.glob(os.path.join(GT_path, seq, '*')))
    seq_path = os.path.join(GT_path.replace('/val', '/noisy_test/val_check'), 'clean', seq) + '.npy'
    gt_images = torch.from_numpy(np.load(seq_path)).to('cuda')
    flows = torch.load(os.path.join(GT_path, seq).replace('DAVIS2Share/val/', 'DAVIS/val_of21/') + '.pt')
    for noise_level in noise_levels:
        seq_path = os.path.join(GT_path.replace('/val', '/noisy_test/val_check'), str(noise_level), seq) + '.npy'
        seq_noisy = torch.from_numpy(np.load(seq_path)).to('cuda')
        seq_noisy = torch.cat([seq_noisy[:1], seq_noisy[:1], seq_noisy, seq_noisy[-1:], seq_noisy[-1:]], dim=0)
        for num, file in enumerate(files):
            if num >= maximum_per_sequence:
                break
            gt_image = gt_images[num:num+1]
            if not num:
                padder = InputPadder(gt_image.shape)
            else:
                prev_occ_mask = load_image(prev_occ_mask_file, range=1.0).cuda()

            if num:
                flow = flows[num-1].unsqueeze(dim=0)
                flow2 = torch.load(os.path.join(GT_path, seq).replace('/val/', '/val_of21/') + '/' + str(num).zfill(5) + '.pt')
                print((flow-flow2).abs().max())
                flow = padder.unpad(flow)
            for model_temp, model_path in zip(models, model_paths):
                with torch.no_grad():
                    N,C,H,W = seq_noisy.shape

                    noisyframe = torch.reshape(seq_noisy[num:num + 5], (1, 15, H, W))

                    sh_im = noisyframe.size()
                    noise_std = torch.FloatTensor([float(noise_level) / 255.0]).to('cuda')
                    noise_std = noise_std.expand((1, 1, H, W))

                    expanded_h = sh_im[-2] % 16
                    if expanded_h:
                        expanded_h = 16 - expanded_h
                    expanded_w = sh_im[-1] % 16
                    if expanded_w:
                        expanded_w = 16 - expanded_w
                    padexp = (0, expanded_w, 0, expanded_h)
                    noisyframe = torch.nn.functional.pad(input=noisyframe, pad=padexp, mode='reflect')
                    noise_std = torch.nn.functional.pad(input=noise_std, pad=padexp, mode='reflect')

                    result_image = model_temp(noisyframe, noise_std)
                    if expanded_h:
                        result_image = result_image[:, :, :-expanded_h, :]
                    if expanded_w:
                        result_image = result_image[:, :, :, :-expanded_w]
                    result_image = torch.clamp(result_image, 0, 1)

                    # SFD
                    result_image_oneFrame = model_temp(torch.cat([noisyframe[:, 6:9] for _ in range(5)], dim=1), noise_std)
                    if expanded_h:
                        result_image_oneFrame = result_image_oneFrame[:, :, :-expanded_h, :]
                    if expanded_w:
                        result_image_oneFrame = result_image_oneFrame[:, :, :, :-expanded_w]
                    result_image_oneFrame = torch.clamp(result_image_oneFrame, 0, 1)

                    if num:
                        # Warping error
                        prev_result = previous_denoised_dict[model_path][noise_level]
                        prev_result_warp = warp(prev_result.cuda(), flow.cuda())
                        results[model_path][noise_level][seq]['WE'].add(compute_WE(prev_result_warp, result_image.cuda(), prev_occ_mask).detach().item())

                    results[model_path][noise_level][seq]['PSNR'].add(PSNR(gt_image, result_image).detach().item())
                    results[model_path][noise_level][seq]['SSIM'].add(SSIM(gt_image, result_image).detach().item())
                    results[model_path][noise_level][seq]['LPIPS'].add(LPIPS((gt_image-0.5)*2, (result_image-0.5)*2).detach().item())
                    PSNR.reset()
                    SSIM.reset()

                    results[model_path][noise_level][seq]['SFD'].add(results[model_path][noise_level][seq]['PSNR'].get_last() - PSNR(gt_image, result_image_oneFrame).detach().item())
                    PSNR.reset()

                    previous_denoised_dict[model_path][noise_level] = result_image

            prev_occ_mask_file = file.replace('/val/', '/val_occlusion/').replace('.jpg', '.png')

    print(seq, end='\t')
    for metric in results[model_path][noise_level][seq]:
        print(metric, end='\t')
    for model_path in model_paths:
        print(f"\n{model_path}")
        for noise_level in noise_levels:
            print(noise_level, end='\t')
            for metric in results[model_path][noise_level][seq]:
                results[model_path][noise_level]['avg'][metric].add(results[model_path][noise_level][seq][metric].avg())
                print("{}".format(results[model_path][noise_level][seq][metric].avg().round(4)), end='\t')
            print()
    print("\n")


print("AVG", end='\t')
for metric in results[model_path][noise_level]['avg']:
    print(metric, end='\t')
for model_path in model_paths:
    print(f"\n{model_path}")
    for noise_level in noise_levels:
        print(noise_level, end='\t')
        for metric in results[model_path][noise_level]['avg']:
            print("{}".format(results[model_path][noise_level]['avg'][metric].avg().round(4)), end='\t')
        print()