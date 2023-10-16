## TO DO
 * Plug VAE sampling to WGAN create_real_samples func
 * Use distribution of sim genomes in PCA as stop metric for WGAN


## TO TEST
 * Try WGAN-GP
 * Like, all the other models mentioned in the manuscript
 * Split genome with no max size, play around with min size / min dist between hotspots
 * Sleep the VAEs to reduce latent space (!! watch out for mode searching !!)


## WHAT I WANT TO TRY
 * Take ~1K top hotspots, with 10-100kb sparsity around them
 * Split with no maxsize
 * VAE 100x with sleeping, on 10?% chunks to optimise hyperparams (A LOT to try)
 * WGAN-GP (with VAE sampling) on ~10 chunks to see if works + approx training resources needed
 * Scale up
 * Try multi chr?
