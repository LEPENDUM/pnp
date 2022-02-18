from __future__ import annotations
from typing import TYPE_CHECKING, Callable, Optional
if TYPE_CHECKING:
	from torch import Tensor
	from pnpcore import Solver
	from argparse import Namespace

from utils.filenames import DirectoryStructure
from utils.logger_utils import logger_info
from utils.image_utils import get_image_paths, imread, imsave, tensor4_to_np3, f32_to_uint8, calculate_ssim, calculate_psnr
import utils.argparse_utils as argutils
import torch
import os
import logging
from collections import OrderedDict



def arguments_list():
	return [
		argutils.Argument(
			name='res_dir', short_name='rd', type=str, default='results',
			help='Root directory for saving the results.'),
		argutils.Argument(
			name='gt_dir', short_name='gtd', type=argutils.check_isdir, default='./testsets',
			help='Root directory of the ground truth dataset for testing (does not include the dataset subfolder).'),
		argutils.Argument(
			name='dataset', short_name='ds', type=str, required=True,
			help='Name of the test dataset (should match with a subfolder name in the gt_dir directory).'),
		argutils.Argument(
			name='save_inputs', short_name='sinp', type=argutils.check_binary, default=1,
			help='Either 0 or 1:\n1 -> save input images given to the pnp reconstruction algorithm.'),
		argutils.Argument(
			name='save_init', short_name='sini', type=argutils.check_binary, default=1,
			help='Either 0 or 1:\n1 -> save initial estimate given to the pnp reconstruction algorithm.'),
		argutils.Argument(
			name='save_rec', short_name='srec', type=argutils.check_binary, default=1,
			help='Either 0 or 1:\n1 -> save reconstructed images by the pnp reconstruction algorithm.'),
		argutils.Argument(
			name='save_rec_rate', short_name='srecr', type=int, default=-1,
			help='Save result every save_rec_rate iterations. Use negative value not to save any intermediate result'),
		argutils.Argument(
			name='num_iterations', short_name='nit', type=argutils.check_nneg_int, required=True,
			help='Maximum number of iterations of the pnp algorithm.'),
	]


def run(initializer: Callable[[dict], Tensor], solver: Solver, args: Namespace):

	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	torch.cuda.empty_cache()
	task = solver.task
	for _, v in solver.named_parameters():
		v.requires_grad = False
	solver.to(device)

	init_name = args.init_name if 'init_name' in args else ''
	ds = DirectoryStructure(args.res_dir, solver, args.dataset, init_name)
	in_dir = ds.input_directory()
	out_dir = ds.output_directory(args.num_iterations)
	init_dir = ds.init_directory()

	if not os.path.exists(out_dir):
		os.makedirs(out_dir)
	if args.save_inputs and not os.path.exists(in_dir):
		os.makedirs(in_dir)
	if args.save_init and not os.path.exists(init_dir):
		os.makedirs(init_dir)

	logger_name = os.path.join(out_dir, 'results.log')
	logger_info(logger_name, log_path=logger_name)
	logger = logging.getLogger(logger_name)

	args_str = ', '.join([item[0] + '=' + str(item[1]) for item in args.__dict__.items()])
	logger.info(f'\nTask: {task.name()}\nArguments:\n{args_str}')

	test_results = OrderedDict()
	test_results['psnr'] = []
	test_results['ssim'] = []

	L_paths = get_image_paths(os.path.join(args.gt_dir, args.dataset))

	for idx, img in enumerate(L_paths):
		img_name, ext = os.path.splitext(os.path.basename(img))
		img_gt = imread(img)
		img_gt = task.reformat_ground_truth(img_gt)
		img_gt_np = f32_to_uint8(tensor4_to_np3(img_gt))
		# n_channels = img_gt.shape[1]

		inputs = task.generate_inputs(img_gt.to(device))
		#inputs = task.generate_inputs(img_gt)
		if args.save_inputs:
			task.set_inputs(inputs)
			task.save_inputs(in_dir, img_name, ext)

		initialisation = initializer(inputs)
		if args.save_init:
			imsave(initialisation, os.path.join(init_dir, img_name + ext))

		solver_gen = solver(inputs, initialisation, args.num_iterations)
		for img_rec in solver_gen:
			if args.save_rec_rate > 0 and (solver.current_iteration - 1) % args.save_rec_rate == 0:
				imsave(img_rec, os.path.join(out_dir, img_name + f'_it{solver.current_iteration}' + ext))

		img_rec_np = f32_to_uint8(tensor4_to_np3(img_rec))

		psnr = calculate_psnr(img_rec_np, img_gt_np)
		ssim = calculate_ssim(img_rec_np, img_gt_np)

		test_results['psnr'].append(psnr)
		test_results['ssim'].append(ssim)
		logger.info('{:->4d}--> {:>10s} PSNR: {:.2f}dB'.format(idx + 1, img_name + ext, psnr))

		if args.save_rec:
			imsave(img_rec, os.path.join(out_dir, img_name + ext))

		# if n_channels == 3:
		# 	img_E_y = util.rgb2ycbcr(img_E, only_y=True)
		# 	img_H_y = util.rgb2ycbcr(img_H, only_y=True)
		# 	psnr_y = util.calculate_psnr(img_E_y, img_H_y)
		# 	test_results['psnr_y'].append(psnr_y)

	# --------------------------------
	# Average PSNR for all images
	# --------------------------------
	ave_psnr = sum(test_results['psnr']) / len(test_results['psnr'])
	logger.info('------> Average PSNR(RGB) of ({}): {:.2f} dB'.format(args.dataset, ave_psnr))

	# if n_channels == 3:
	# 	ave_psnr_y = sum(test_results['psnr_y']) / len(test_results['psnr_y'])
	# 	logger.info('------> Average PSNR(Y) of ({}): {:.2f} dB'.format(testset_name, ave_psnr_y))
