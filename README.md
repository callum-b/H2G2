# H2G2
Haplotypic Human Genome Generator

This is the main product of my PhD project: a way to generate haplotypic human genomic data, based on a reference dataset.

## Project
The full details of this project are available as a biorXiv preprint. Read it here: *LINK COMING SOON*

## How to use
Here the main scripts for data pre-processing and building ANN models are provided. There will be manual transofrmations to tie these functions together, as well as decisions to make on your end about some of the hyperparameters and criteria for the project.
Currently the steps are:
 - determine mutation segmentation loci (I use recombination hotspots form Halldorsson, et al. 2019)
 - split .vcf file according to the loci
 - split genotypes into haplotypes for sections you are interested in
 - autoencode these sections (I recommend using VAE with sigmoid activation in layers, linear latent space)
 - train WGAN on multiple encoded subsections until generator provides realistic and diverse samples (I used checkpopints every 10k training epochs, decoded the samples generated at these checkpoints then compared those simulated samples to the reference ones by eye using PCA)
 - congratulations! You now have a haplotypic human genome generator!

## Pipeline
I am hoping one day to wrap this all in a snakemake pipeline, but that's still a way away. In any case, I think having the code visible and accessible is important in case you want to tweak anything.

## License
This code is released under CECILL 2.1. It is therefore sharable, reusable and modifiable, as long as this source is credited as the original author.
