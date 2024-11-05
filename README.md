# H2G2
Haplotypic Human Genome Generator

This is the main product of my PhD project: a way to generate haplotypic human genomic data, based on a reference dataset.

## Project
The full details of this project are available as a biorXiv preprint. Read it here: https://www.biorxiv.org/content/10.1101/2023.12.08.570767v1.full

## How to use 
The scripts included here can be ran independantly. The main steps are:
 - determine mutation segmentation loci (I use recombination hotspots form Halldorsson, et al. 2019)
 - split .vcf file according to the loci
 - split genotypes into haplotypes for sections you are interested in
 - autoencode these sections (I recommend using VAE with sigmoid activation in layers, linear latent space)
 - train WGAN on multiple encoded subsections until generator provides realistic and diverse samples (I used checkpopints every 10k training epochs, decoded the samples generated at these checkpoints then compared those simulated samples to the reference ones by eye using PCA)
 - congratulations! You now have a haplotypic human genome generator!

However, I also included a Snakemake pipeline which can run all these steps in succession.  If you have never used Snakemake before at all, you might want to go over the basic principles on their website: https://snakemake.readthedocs.io/en/stable/

### Installing Snakemake and its environment
After cloning this github repo, you'll need to create the mamba environment required to run the scripts. Micromamba installation instructions are found on its host doc site: https://mamba.readthedocs.io/en/latest/installation/micromamba-installation.html

Use micromamba to create the environment described in the config file: 

    `micromamba create -f h2g2.yaml`
    
Activate the mamba environment you just created:

    `micromamba activate h2g2`
Use pip to install the packages that were not instal
led via micromamba:

    `pip3 install pyranges tensorflow==2.12.0`
    
You should now be ready to run the snakemake pipeline.

### Snakemake rules
The Snakemake pipeline is defined in the `Snakefile`. It contains all the individual rules used to run each software, as well as "big picture" rules that will run a batch of them at a time.

As long as you are running the pipeline with default parameters, you will not need to modify anything in the `config.yaml` configuration file. 

First, run `snakemake --cores=4 collect_and_split_data` which will create haplotypic vcf files for each subsection along chromosome 1.

Once you have those, you should create the subdirectory `DATA/VCF/chr1/chunks_1Mb`, and populate it with the files for the Mb of data processed for the publication: `cp DATA/VCF/chr1/2??????_*_haplo.vcf DATA/VCF/chr1/chunks_1Mb/`. To save disk space, you can then gzip all the unused vcf files using parallel: `parallel gzip ::: DATA/VCF/chr1/*vcf &`

Then you can run `snakemake --cores=4 wgan_some_sections`, which will train a WGAN model on those 1Mb of haplotypic vcf files. Thsi training step is quite long, and is prone to failure. I recommend looking at a graph of the the critic and generator loss over the training steps to select a checkpoint that seems appropriate, or re-running this training if it seems unsatisfactory.

That will also create 1000 (by default) simulated haplotypic genomes, which can be used to test the realism of the generated samples.

`snakemake --cores=4 paint_part_chromosome_autodetect` will prepare the necessary datafiles for chromopainter and run it on them, which will allow you to view the reconstructed ancestry of the simulated genomes.

All other evaluation metrics were determined using classical methods like PCA or Hamming distances, which can be implemented quite easily in python.

## License
This code is released under CECILL 2.1. It is therefore sharable, reusable and modifiable, as long as this source is credited as the original author.
Copyright CNRS 2023

This software is governed by the CeCILL  license under French law and abiding
by the rules of distribution of free software. You can use, modify and/ or
redistribute the software under the terms of the CeCILL license as circulated
by CEA, CNRS and INRIA at the following URL:
http://www.cecill.info/index.en.html
As a counterpart to the access to the source code and  rights to copy, modify
and redistribute granted by the license, users are provided only with a limited
warranty  and the software's author,the holder of the economic rights, and the
successive licensors have only limited liability.
In this respect, the user's attention is drawn to the risks associated with
loading, using, modifying and/or developing or reproducing the software by the
user in light of its specific status of free software, that may mean  that it
is complicated to manipulate, and that also therefore means  that it is
reserved for developers  and  experienced professionals having in-depth
computer knowledge. Users are therefore encouraged to load and test the
software's suitability as regards their requirements in conditions enabling
the security of their systems and/or data to be ensured and, more generally,
to use and operate it in the same conditions as regards security.
The fact that you are presently reading this means that you have had knowledge
of the CeCILL license and that you accept its terms.


