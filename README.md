
# pnp: Plug-And-Play algorithms for inverse problems

This repository implements the preconditioned plug-and-play method described in the following paper:

M. Le Pendu and C. Guillemot, "Preconditioned Plug-and-Play ADMM with Locally Adjustable Denoiser for Image Restoration" [[arXiv]](https://arxiv.org/pdf/2110.00493.pdf)

The code was tested for Python 3.7 and PyTorch 1.7.0


## Abstract

Plug-and-Play optimization recently emerged as a powerful technique for solving inverse problems by plugging a denoiser into a classical optimization algorithm.
The denoiser accounts for the regularization and therefore implicitly determines the prior knowledge on the data, hence replacing typical handcrafted priors.
In this paper, we extend the concept of plug-and-play optimization to use denoisers that can be parameterized for non-constant noise variance.
In that aim, we introduce a preconditioning of the ADMM algorithm, which mathematically justifies the use of such an adjustable denoiser.
We additionally propose a procedure for training a convolutional neural network for high quality non-blind image denoising that also allows for pixel-wise control of the noise standard deviation.
We show that our pixel-wise adjustable denoiser, along with a suitable preconditioning strategy, can further improve the plug-and-play ADMM approach for several applications,
including image completion, interpolation, demosaicing and Poisson denoising.




## Usage examples
- Pretrained models are not included in this repository. See /model_zoo folders for download instructions.
- Default settings are configured as described in the paper. See --help for all options (e.g. number of iterations, denoising level at first and last iteration, ... ) 
- Use the option --dataset to change the test dataset name (by default, set5 is used for interpolation, completion and Poisson denoising, and kodak is used for demosaicing)

### Interpolation (using set5 dataset)
```bash
    python Interpolation.py -sf 2 -p	# 2x upsampling test, with preconditioning.
	python Interpolation.py -sf 2		# 2x upsampling test, without preconditioning.
	python Interpolation.py -sf 4 -p	# 4x upsampling test, with preconditioning.
	python Interpolation.py -sf 4		# 4x upsampling test, without preconditioning.
```

### Completion (using set5 dataset)

```bash
	python Completion.py -r 20 -p	# completion from 20% of known pixels, with preconditioning.
	python Completion.py -r 20		# completion from 20% of known pixels, without preconditioning.
	python Completion.py -r 10 -p	# completion from 10% of known pixels, with preconditioning.
	python Completion.py -r 10		# completion from 10% of known pixels, without preconditioning.
```

### Demosaicing (using kodak dataset)
```bash
	# Noise-free Demosaicing
    python Demosaic.py -p			# with preconditioning, using generic denoiser DRUNET-var-RGB.
	python Demosaic.py -p -dem		# with preconditioning, using specific denoiser DRUNET-dem.
	python Demosaic.py 				# without preconditioning, using DRUNET-cst denoiser.

	# Joint Demosaicing-Denoising
    python Demosaic.py -sigma_noise_8b=20 -p		# noise level 20/255, with preconditioning, using generic denoiser DRUNET-var-RGB.
	python Demosaic.py -sigma_noise_8b=20 -p -dem	# noise level 20/255, with preconditioning, using specific denoiser DRUNET-dem.
	python Demosaic.py -sigma_noise_8b=20			# noise level 20/255, without preconditioning.
```	

### Poisson denoising (using set5 dataset)
```bash	
    python PoissonDenoise.py -peak=255 -p				# Denoising for low Poisson noise level (peak=255), with preconditioning.
	python PoissonDenoise.py -peak=255					# Denoising for low Poisson noise level (peak=255), without preconditioning.
	python PoissonDenoise.py -peak=1 -init_anscombe -p	# Denoising for very high Poisson noise level (peak=1), initialised with Anscombe method, with preconditioning.
	python PoissonDenoise.py -peak=1 -init_anscombe		# Denoising for very high Poisson noise level (peak=1), initialised with Anscombe method, without preconditioning.
```






## Citation
If you find the code helpful in your resarch or work, please cite the following paper:

```BibTex
@article{lependu2021preconditioned,
  title={Preconditioned Plug-and-Play ADMM with Locally Adjustable Denoiser for Image Restoration},
  author={Le Pendu, Mikael and Guillemot, Christine},
  journal={arXiv preprint arXiv:2110.00493},
  year={2021}
}
```