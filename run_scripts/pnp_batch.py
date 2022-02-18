from __future__ import annotations
from typing import TYPE_CHECKING, Callable
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

		argutils.Argument(
			name='batch_size', short_name='bs', type=argutils.check_pos_int, default=1,
			help='Number of images in a batch.'),
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

	num_imgs = len(L_paths)
	start_idx = 0
	while start_idx < num_imgs:
		end_idx = start_idx + min(args.batch_size, num_imgs - start_idx)
		enum_batch = list(enumerate(L_paths[start_idx:end_idx]))

		# Load a batch of images
		img_names = [''] * (end_idx - start_idx)
		for idx, img in enum_batch:
			img_name, ext = os.path.splitext(os.path.basename(img))
			img_names[idx] = img_name
			img_gt = imread(img)
			img_gt = task.reformat_ground_truth(img_gt)
			if idx == 0:
				img_gt_batch = torch.empty(end_idx - start_idx, img_gt.shape[1], img_gt.shape[2], img_gt.shape[3])
			if img_gt.shape[1:] != img_gt_batch.shape[1:]:
				raise RuntimeError(
					f'All the images in the batch should have the same size. '
					f'Size of image \'{img_name}\': {list(img_gt.shape[1:])}. '
					f'Expected size: {list(img_gt_batch.shape[1:])}.')
			img_gt_batch[idx:idx + 1] = img_gt

		del img_gt

		# Generate degraded input images
		inputs = task.generate_inputs(img_gt_batch.to(device))
		if args.save_inputs:
			task.set_inputs(inputs)
			task.save_inputs(in_dir, img_names, ext)

		# Compute initial estimate
		initialisation = initializer(inputs)
		if args.save_init:
			for idx, img in enum_batch:
				imsave(initialisation[idx:idx + 1], os.path.join(init_dir, img_names[idx] + ext))

		# Solve the inverse problem
		solver_gen = solver(inputs, initialisation, args.num_iterations)
		for img_rec in solver_gen:
			if args.save_rec_rate > 0 and (solver.current_iteration - 1) % args.save_rec_rate == 0:
				for idx, img in enum_batch:
					imsave(img_rec[idx:idx + 1], os.path.join(out_dir, img_names[idx] + f'_it{solver.current_iteration}' + ext))

		# Compute metrics and save results
		for idx, img in enum_batch:
			img_gt_np = f32_to_uint8(tensor4_to_np3(img_gt_batch[idx:idx+1]))
			img_rec_np = f32_to_uint8(tensor4_to_np3(img_rec[idx:idx+1]))
			psnr = calculate_psnr(img_rec_np, img_gt_np)
			ssim = calculate_ssim(img_rec_np, img_gt_np)

			test_results['psnr'].append(psnr)
			test_results['ssim'].append(ssim)
			logger.info('{:->4d}--> {:>10s} PSNR: {:.2f}dB'.format(start_idx + idx + 1, img_names[idx] + ext, psnr))

			if args.save_rec:
				imsave(img_rec[idx:idx+1], os.path.join(out_dir, img_names[idx] + ext))

		start_idx = end_idx

	# --------------------------------
	# Average PSNR for all images
	# --------------------------------
	ave_psnr = sum(test_results['psnr']) / len(test_results['psnr'])
	logger.info('------> Average PSNR(RGB) of ({}): {:.2f} dB'.format(args.dataset, ave_psnr))
