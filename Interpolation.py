import run_scripts.pnp_batch as pnp_script
from algorithms.admm import ADMMSolver
from tasks.recover_masked_data import RecoverMaskedData, GridMaskGenerator, MaskPrecoCreator, InitImageInterp
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
            name='sampling_factor', short_name='sf', type=int, default=2,
            help='subsampling factor of the input data to be interpolated.'),
        argutils.Argument(
            name='use_preco', short_name='p', action='store_true',
            help='Use preconditioning.'),
        argutils.Argument(
            name='p_max', short_name='p_max', type=float, default=10.0,
            help='Maximum preconditionning value (10 by default).'),
        argutils.Argument(
            name='sigma_blur_last_iter', short_name='sbli', type=float, default=0.4,
            help='Sigma parameter for the gaussian blurring of the preconditionner '
                 '(the parameter increases over the iterations up to the given value for the last iteration).'),
    ]


def parse_arguments():
    args_list = arguments_list() + pnp_script.arguments_list()
    argutils.overwrite_default(args_list, num_iterations=None, dataset='set5')
    parser = argutils.make_parser(args_list, formatter_class=argparse.RawTextHelpFormatter)
    return parser.parse_args()


def main():
    args = parse_arguments()

    # Define task:
    mask_gen = GridMaskGenerator(sampling_factor=args.sampling_factor)
    task = RecoverMaskedData(mask_gen)

    # Define preconditioner:
    if args.use_preco:
        preconditioner = DiagonalPreconditioner()
        preconditioner.set_preco_creator(
            MaskPrecoCreator(p_max=args.p_max, sigma_blur_last_iter=args.sigma_blur_last_iter)
        )
    else:
        preconditioner = no_preco

    # Define regularizer:
    if args.use_preco:
        drunet_type = 'var'  # cst, var, var-rgb, dem
    else:
        drunet_type = 'cst'
    denoiser_factory = DrunetFactory(drunet_type)
    denoiser = ImageModuleWrapper(denoiser_factory.denoiser_non_iid(), run_strategy=SplitRun(), x8_augment=True)
    regularizer = DenoisingRegularizer(denoiser, denoiser_factory.denoiser_caller())

    # Define initialization:
    initializer = InitImageInterp('bicubic')

    # Configure the solver:
    solver = ADMMSolver(sigma_den_0=50 / 255, sigma_den_N=1 / 255, task=task, regularizer=regularizer,
                        preconditioner=preconditioner)

    if args.num_iterations is None:  # set default number of iterations from the paper
        args.num_iterations = 6 if args.sampling_factor <= 2 else 10
    pnp_script.run(initializer, solver, args)


if __name__ == "__main__":
    main()
