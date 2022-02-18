import run_scripts.pnp_batch as pnp_script
from algorithms.admm import ADMMSolver
from tasks.recover_masked_data import RecoverMaskedData, BayerMaskGenerator, MaskPrecoCreator, InitDemosaic
from pnpcore.preconditioner import DiagonalPreconditioner
from pnpcore import no_preco
from regularizers.denoise_reg import DenoisingRegularizer
from models.drunet import DrunetFactory
from models.image_module import ImageModuleWrapper, SplitRun

import utils.argparse_utils as argutils
import argparse


def arguments_list():
    return [
        argutils.Argument(
            name='sigma_noise_8b', short_name='sn8b', type=float, default=0,
            help='Standard deviation of the Gaussian noise added to the RAW data '
                 '(assuming pixel data in the range [0,255]).'),
        argutils.Argument(
            name='use_preco', short_name='p', action='store_true',
            help='Use preconditioning.'),
        argutils.Argument(
            name='use_drunet_dem', short_name='dem', action='store_true',
            help='Use specialized locally adjustable drunet denoiser for demosaicing preconditioning patterns.'),
        argutils.Argument(
            name='p_max', short_name='p_max', type=float, default=10.0,
            help='Maximum preconditionning value (10 by default).'),
        argutils.Argument(
            name='sigma_blur_last_iter', short_name='sbli', type=float,
            help='Sigma parameter for the gaussian blurring of the preconditionner '
                 '(the parameter increases over the iterations up to the given value for the last iteration).'),
    ]


def parse_arguments():
    args_list = arguments_list() + pnp_script.arguments_list()
    argutils.overwrite_default(args_list, num_iterations=None, dataset='kodak')
    parser = argutils.make_parser(args_list, formatter_class=argparse.RawTextHelpFormatter)
    return parser.parse_args()


def main():
    args = parse_arguments()

    # Define task:
    mask_gen = BayerMaskGenerator('rggb')
    task = RecoverMaskedData(mask_gen, sigma_noise=args.sigma_noise_8b/255)

    # Define preconditioner:
    if args.use_preco:
        if args.sigma_blur_last_iter is None:
            args.sigma_blur_last_iter = 0. if args.use_drunet_dem else 0.3
        preconditioner = DiagonalPreconditioner()
        preconditioner.set_preco_creator(
            MaskPrecoCreator(p_max=args.p_max, sigma_blur_last_iter=args.sigma_blur_last_iter)
        )
    else:
        preconditioner = no_preco

    # Define regularizer:
    if args.use_preco:
        drunet_type = 'dem' if args.use_drunet_dem else 'var-rgb'  # cst, var, var-rgb, dem
    else:
        drunet_type = 'cst'
    denoiser_factory = DrunetFactory(drunet_type)
    denoiser = ImageModuleWrapper(denoiser_factory.denoiser_non_iid(), run_strategy=SplitRun(), x8_augment=True)
    regularizer = DenoisingRegularizer(denoiser, denoiser_factory.denoiser_caller())

    # Configure the solver:
    solver = ADMMSolver(sigma_den_0=50 / 255, sigma_den_N=max(1, args.sigma_noise_8b) / 255, task=task,
                        regularizer=regularizer, preconditioner=preconditioner)

    # Define initialization:
    initializer = InitDemosaic()

    if args.num_iterations is None:  # set default number of iterations from the paper
        if args.use_preco:
            args.num_iterations = 10
        elif args.sigma_noise_8b > 0:
            args.num_iterations = 16
        else:
            args.num_iterations = 40
    pnp_script.run(initializer, solver, args)


if __name__ == "__main__":
    main()
