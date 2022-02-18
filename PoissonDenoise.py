import run_scripts.pnp_batch as pnp_script
from algorithms.admm import ADMMSolver
from tasks.poisson import PoissonDenoise, PoissonPrecoCreator, init_noisy, InitAnscombeDenoise
from pnpcore.preconditioner import DiagonalPreconditioner
from pnpcore import no_preco
from regularizers.denoise_reg import DenoisingRegularizer
from models.drunet import DrunetFactory
from models.image_module import ImageModuleWrapper, SplitRun

import utils.argparse_utils as argutils
import argparse

import math


def arguments_list():
    return [
        argutils.Argument(
            name='peak', type=argutils.check_pos_numeric, default=255 / 8,
            help='Peak value for Poisson noise (small value -> high noise level).'),
        argutils.Argument(
            name='use_preco', short_name='p', action='store_true',
            help='Use preconditioning.'),
        argutils.Argument(
            name='init_anscombe', action='store_true',
            help='Use Gaussian denoising with Anscombe transform for initialisation.'),
    ]


def parse_arguments():
    args_list = arguments_list() + pnp_script.arguments_list()
    argutils.overwrite_default(args_list, num_iterations=6, dataset='set5')
    parser = argutils.make_parser(args_list, formatter_class=argparse.RawTextHelpFormatter)
    return parser.parse_args()


def main():
    args = parse_arguments()

    # Define task:
    task = PoissonDenoise(args.peak)

    # Define preconditioner:
    if args.use_preco:
        preconditioner = DiagonalPreconditioner()
        # Defines how preconditioner is created and updated. Here, we the update is only performed if the intilisation
        # is the noisy image (i.e. no Anscombe initialization).
        preconditioner.set_preco_creator(PoissonPrecoCreator(do_update=not args.init_anscombe))
        sig_0 = 1
        sig_N = 1
    else:
        preconditioner = no_preco
        sig_0 = math.sqrt(args.peak)
        sig_N = math.sqrt(args.peak)/2

    # Define regularizer:
    if args.use_preco:
        drunet_type = 'var-rgb'  # cst, var, var-rgb, dem
    else:
        drunet_type = 'cst'
    denoiser_factory = DrunetFactory(drunet_type)
    denoiser = ImageModuleWrapper(denoiser_factory.denoiser_non_iid(), run_strategy=SplitRun(), x8_augment=True)
    regularizer = DenoisingRegularizer(denoiser, denoiser_factory.denoiser_caller())

    # Define initialization:
    if args.init_anscombe:
        args.init_name = 'Anscombe'
        initializer = InitAnscombeDenoise(
            ImageModuleWrapper(denoiser_factory.denoiser_iid(), run_strategy=SplitRun(), x8_augment=args.use_x8)
        )
    else:
        initializer = init_noisy

    # Configure the solver:
    solver = ADMMSolver(sigma_den_0=sig_0, sigma_den_N=sig_N, task=task, regularizer=regularizer,
                        preconditioner=preconditioner)

    pnp_script.run(initializer, solver, args)


if __name__ == "__main__":
    main()

