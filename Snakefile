"""

The Haplotypic Human Genome Generator (H2G2) Snakemake pipeline
v0.1

This pipeline is designed to run the scripts developped by C. BURNARD to simulate novel Human genomic data. See the article here: https://www.biorxiv.org/content/10.1101/2023.12.08.570767v1
This Snakefile was most likely obtained form this github repo: https://github.com/callum-b/H2G2 , but if you got it from somewhere else, I'd be curious to know where that was.

"""


### BASIC CHECKS & IMPORTS ###
configfile: "config.yaml"

from multiprocessing import cpu_count
import pandas as pd
import glob
import re
import os

## import custom scripts (the ones from github)
from SCRIPTS import datapreprocessing


### GLOBAL VARIABLES ###
MAX_CORES = cpu_count()
working_dir = os.getcwd()

wildcard_constraints:
    chrom="chr\d{1,2}",
    section="\d+_\d+"

### FUNCTIONS ###
def get_sections_wgan_encoded_fullchr(wildcards):
    filelist = glob.glob("DATA/VCF/"+wildcards.chrom+"/*[0-9].vcf???")
    if filelist:
        return sorted(set(x.replace("VCF", "ENCODED").replace(".vcf", "_haplo_encoded_sampling.csv").replace(".gz", "") 
            for x in filelist))
    else:
        return sorted(set(x.replace("VCF", "ENCODED").replace(".vcf", "_encoded_sampling.csv").replace(".gz", "") 
            for x in glob.glob("DATA/VCF/"+wildcards.chrom+"/*haplo.vcf???")))

def get_sections_wgan_encoded_subdir(wildcards):
    name=config.get("wgan_subdir", "")
    if name:
        return ["DATA/ENCODED/"+wildcards.chrom+"/" + name + "/" + x + "_haplo_encoded_sampling.csv" for x in config["wgan_subdir_sections"]]
    else:
        return [ "DATA/ENCODED/chr1/chunks_1Mb/" + x + "_haplo_encoded_sampling.csv" for x in ["2037636_2099376", "2099376_2185247", "2185247_2285807", "2285807_2320833", "2320833_2367711", "2367711_2413056", "2413056_2470342", 
                "2470342_2536586", "2536586_2802297", "2802297_2836569", "2872483_2917116", "2917116_2960624", "2960624_3009020"]]


### BIG PICTURE RULES ###

rule collect_and_split_data:
    input:
        expand("DATA/VCF/{chromosome}/ALL_split_haplo.txt", chromosome=config.get("chromosomes_to_process", ["chr1"]))

rule wgan_all_sections:
    input:
        expand("DATA/MODELS/{chromosome}/WGAN.weights.h5", chromosome=config.get("chromosomes_to_process", ["chr1"]))

rule wgan_some_sections:
    input:
        expand("DATA/MODELS/{chromosome}/chunks_1Mb/WGAN.weights.h5", chromosome=config.get("chromosomes_to_process", ["chr1"]))

rule paint_full_chromosome_autodetect:
    input:
        expand("DATA/CHROMOPAINTER/{chromosome}/{section}/out_{section}_vae.samples.out", 
            chromosome=config.get("chromosomes_to_process", ["chr1"]), 
            section=[x.split("/")[-1].replace("_haplo_encoded_sampling_wgan_gen.csv", "") for x in glob.glob("DATA/GENERATED/chr1/*_haplo_encoded_sampling_wgan_gen.csv")] 
             )

rule paint_part_chromosome_autodetect:
    input:
        expand("DATA/CHROMOPAINTER/{chromosome}/{section}_vae/out_{section}_vae.samples.out", 
            chromosome=config.get("chromosomes_to_process", ["chr1"]), 
            section=[x.split("/")[-1].replace("_haplo_encoded_sampling_wgan_gen.csv", "") for x in glob.glob("DATA/GENERATED/chr1/chunks_1Mb/*_haplo_encoded_sampling_wgan_gen.csv")] 
             )

### INDIVIDUAL RULES ###

## PUBLIC DATA PREPROCESSING ##

rule download_genomes_lowygallego:
    output:
        "DATA/VCF/ALL_{chrom}_shapeit2_integrated_snvindels_v2a_27022019_GRCh38_phased.vcf.gz"
    log:
        "snakemake_logs/download_genomes_lowygallego/{chrom}.log"
    shell:
        "touch {output}; wget -O {output} http://ftp.1000genomes.ebi.ac.uk/vol1/ftp/data_collections/1000_genomes_project/release/20190312_biallelic_SNV_and_INDEL/ALL.{wildcards.chrom}.shapeit2_integrated_snvindels_v2a_27022019.GRCh38.phased.vcf.gz "

rule recomb_halldorsson_to_filt_bed:
    input:
        "DATA/BED/aau1043_datas3.gz" # can be obtained at https://www.science.org/doi/suppl/10.1126/science.aau1043/suppl_file/aau1043_datas3.gz (can't use wget though unfortunately :/ )
    output:
        "DATA/BED/genetic_map_sexavg_nosmall.bdg"
    log:
        "snakemake_logs/recomb_halldorsson_to_filt_bed.log"
    params:
        thrs=config.get("recomb_halldorsson_to_bed_thrs", "5")
    shell:
        "echo 'type=bedGraph' > {output}; gunzip -c {input} | tail -n +9 | awk '{{ if(int($4)>{params.thrs}) {{ print $1, $2, $3, $4}} }}' >> {output}"
        
rule recomb_halldorsson_to_bed:
    input:
        "DATA/BED/aau1043_datas3.gz" # can be obtained at https://www.science.org/doi/suppl/10.1126/science.aau1043/suppl_file/aau1043_datas3.gz (can't use wget though unfortunately :/ )
    output:
        "DATA/BED/genetic_map_sexavg.bdg"
    log:
        "snakemake_logs/recomb_halldorsson_to_bed.log"
    shell:
        "echo 'type=bedGraph' > {output}; gunzip -c {input} | tail -n +9 | awk '{{ print $1, $2, $3, $4}}' >> {output}"

rule remove_neighbour_hotspots:
    input:
        "DATA/BED/genetic_map_sexavg_nosmall.bdg"
    output:
        "DATA/BED/genetic_map_sexavg_hotspots_{chrom}.bdg"
    log:
        "snakemake_logs/remove_neighbour_hotspots_{chrom}.log"
    params:
        dist=config.get("remove_neighbour_hotspots_dist", "10000"),
        thrs=config.get("remove_neighbour_hotspots_thrs", "1")
    run:
        datapreprocessing.recombproba_to_hotspots(input[0], wildcards.chrom, int(params.dist), int(params.thrs)).to_bed(output[0])

rule separate_vcf_hotspots:
    input:
        mutations="DATA/VCF/{data}_{chrom}_shapeit2_integrated_snvindels_v2a_27022019_GRCh38_phased.vcf.gz",
        hotspots="DATA/BED/genetic_map_sexavg_hotspots_{chrom}.bdg"
    output:
        "DATA/VCF/{chrom}/{data}_split.txt"
    log:
        "snakemake_logs/separate_vcf_hotspots_{chrom}_{data}.log"
    params:
        maxn=config.get("separate_vcf_hotspots_maxn", "5000"),
        minn=config.get("separate_vcf_hotspots_minn", "500"),
        thrs=config.get("separate_vcf_hotspots_thrs", "3"),
        keep=config.get("separate_vcf_hotspots_keep", "False")
    run:
        datapreprocessing.split_vcf(input.hotspots, input.mutations, "DATA/VCF/" + wildcards.chrom + "/", int(params.maxn), int(params.minn), int(params.thrs), bool(params.keep), False)
        shell("touch {output}")

rule separate_vcf_hotspots_haplo:
    input:
        mutations="DATA/VCF/{data}_{chrom}_shapeit2_integrated_snvindels_v2a_27022019_GRCh38_phased.vcf.gz",
        hotspots="DATA/BED/genetic_map_sexavg_hotspots_{chrom}.bdg"
    output:
        diplo="DATA/VCF/{chrom}/{data}_split.txt", 
        haplo="DATA/VCF/{chrom}/{data}_split_haplo.txt"
    log:
        "snakemake_logs/separate_vcf_hotspots_haplo_{chrom}_{data}.log"
    params:
        maxn=config.get("separate_vcf_hotspots_haplo_maxn", "5000"),
        minn=config.get("separate_vcf_hotspots_haplo_minn", "500"),
        thrs=config.get("separate_vcf_hotspots_haplo_thrs", "3"),
        keep=config.get("separate_vcf_hotspots_haplo_keep", "False")
    run:
        datapreprocessing.split_vcf(input.hotspots, input.mutations, "DATA/VCF/" + wildcards.chrom + "/", int(params.maxn), int(params.minn), int(params.thrs), bool(params.keep), True)
        shell("touch {output.diplo}")
        shell("touch {output.haplo}")

rule haplo_section_auto:
    input:
        "DATA/VCF/{chrom}/{section}.vcf"
    output:
        "DATA/VCF/{chrom}/{section}_haplo.vcf"
    log:
        "snakemake_logs/haplo_section_{chrom}_{section}.log"
    run:
        datapreprocessing.vcf_to_haplo("{input}", "{output}")

## ENCODING AND DECODING ##

rule autoencode_section:
    input:
        "DATA/VCF/{chrom}/{section}_haplo.vcf"
    output:
        data_enc="DATA/ENCODED/{chrom}/{section}_haplo_encoded.csv",
        data_dec="DATA/ENCODED/{chrom}/{section}_haplo_decoded.vcf",
        model_enc="DATA/MODELS/{chrom}/enc_ae_{section}.keras",
        model_dec="DATA/MODELS/{chrom}/dec_ae_{section}.keras"
    log:
        "snakemake_logs/autoencode_section_{chrom}_{section}.log"
    shell:
        "python3 SCRIPTS/autoencode.py {input} {output.data_enc} {output.data_dec} {output.model_enc} {output.model_dec}"

rule VAE_section:
    input:
        "DATA/VCF/{chrom}/{section}_haplo.vcf"
    output:
        data_enc_samp="DATA/ENCODED/{chrom}/{section}_haplo_encoded_sampling.csv", 
        data_dec_samp="DATA/ENCODED/{chrom}/{section}_haplo_decoded_sampling.csv", 
        data_enc_mean="DATA/ENCODED/{chrom}/{section}_haplo_encoded_mean.csv", 
        data_dec_mean="DATA/ENCODED/{chrom}/{section}_haplo_decoded_mean.csv", 
        data_enc_logvar="DATA/ENCODED/{chrom}/{section}_haplo_encoded_logvar.csv", 
        model_enc="DATA/MODELS/{chrom}/enc_vae_{section}.keras",
        model_dec="DATA/MODELS/{chrom}/dec_vae_{section}.keras"
    log:
        "snakemake_logs/VAE_section_{chrom}_{section}.log"
    shell:
        "python3 SCRIPTS/var_autoencode.py {input} "
        "DATA/ENCODED/{wildcards.chrom}/{wildcards.section}_haplo_encoded_ DATA/ENCODED/{wildcards.chrom}/{wildcards.section}_haplo_decoded_ " ##need to leave these incomplete so script will add sampling, mean and logvar
        "{output.model_enc} {output.model_dec}"

rule decode_encoded:
    input:
        model_dec="DATA/MODELS/{chrom}/dec_{type}_{section}.keras",
        data_enc="DATA/ENCODED/{chrom}/{section}_haplo_encoded_sampling_wgan_gen.csv",
        data_ref="DATA/VCF/{chrom}/{section}_haplo.vcf"
    output:
        data_dec="DATA/VCF/{chrom}/{section}_{type}_haplo_encoded_sampling_wgan_gen_decoded.vcf"
    log:
        "snakemake_logs/decode_generated_{chrom}_{section}_{type}.log"
    shell:
        "python3 SCRIPTS/decode.py {input.model_dec} {input.data_enc} {input.data_ref} {output}"

rule decode_generated:
    input:
        model_dec="DATA/MODELS/{chrom}/dec_{type}_{section}.keras",
        data_enc="DATA/GENERATED/{chrom}/{section}_haplo_encoded_sampling_wgan_gen.csv",
        data_ref="DATA/VCF/{chrom}/{section}_haplo.vcf"
    output:
        data_dec="DATA/GENERATED/{chrom}/{section}_{type}_haplo_encoded_sampling_wgan_gen_decoded.vcf"
    log:
        "snakemake_logs/decode_generated_{chrom}_{section}_{type}.log"
    shell:
        "python3 SCRIPTS/decode.py {input.model_dec} {input.data_enc} {input.data_ref} {output}"

## MANAGING DATA ##

rule symlink_csv:
    input:
        "DATA/{type}/{chrom}/{filename}.csv"
    output:
        "DATA/{type}/{chrom}/{subdir}/{filename}.csv"
    log:
        "snakemake_logs/symlink_csv_{type}_{chrom}_{subdir}_{filename}.log"
    shell:
        "ln -s "+working_dir+"/{input} {output}"

rule symlink_vcf:
    input:
        "DATA/{type}/{chrom}/{filename}.vcf"
    output:
        "DATA/{type}/{chrom}/{subdir}/{filename}.vcf"
    log:
        "snakemake_logs/symlink_vcf_{type}_{chrom}_{subdir}_{filename}.log"
    shell:
        "ln -s "+working_dir+"/{input} {output}"

rule gunzip_vcf:
    input:
        "DATA/VCF/{chrom}/{filename}.vcf.gz"
    output:
        temp("DATA/VCF/{chrom}/{filename}.vcf")
    log:
        "snakemake_logs/gunzip_vcf_{chrom}_{filename}.log"
    shell:
        "gunzip -k {input}"

## TRAINING GENERATOR ##

rule wgan_encoded_fullchr:
    input:
        get_sections_wgan_encoded_fullchr
    output:
        model = "DATA/MODELS/{chrom}/WGAN.weights.h5"
    log:
        "snakemake_logs/wgan_encoded_{chrom}.log"
    params:
        train_steps = config.get("wgan_encoded_fullchr_train_steps", "50000"),
        train_save_steps = config.get("wgan_encoded_fullchr_train_save_steps", "100"),
        train_check_steps = config.get("wgan_encoded_fullchr_train_check_steps", "1000"),
        output_samples = config.get("wgan_encoded_fullchr_output_samples", "1000")
    shell:
        "python3 SCRIPTS/wgan_encoded.py {params.train_steps} {params.train_save_steps} {params.train_check_steps} {params.output_samples} {output.model} {input}"

rule wgan_encoded_subdir:
    input:
        get_sections_wgan_encoded_subdir
    output: 
        model = "DATA/MODELS/{chrom}/{subdir}/WGAN.weights.h5"
    log:
        "snakemake_logs/wgan_encoded_{chrom}_{subdir}.log"
    params:
        train_steps = config.get("wgan_encoded_subdir_train_steps", "50000"),
        train_save_steps = config.get("wgan_encoded_subdir_train_save_steps", "100"),
        train_check_steps = config.get("wgan_encoded_subdir_train_check_steps", "1000"),
        output_samples = config.get("wgan_encoded_subdir_output_samples", "1000")
    shell:
        "python3 SCRIPTS/wgan_encoded.py {params.train_steps} {params.train_save_steps} {params.train_check_steps} {params.output_samples} {output.model} {input}"

## ANALYSING RESULTS ##

rule chromopainter_prep:
    input:
        data_dec="DATA/GENERATED/{chrom}/{section}_{type}_haplo_encoded_sampling_wgan_gen_decoded.vcf",
        data_ref="DATA/VCF/{chrom}/{section}_haplo.vcf"
    output:
        donor="DATA/CHROMOPAINTER/{chrom}/{section}_{type}/donor.txt",
        donor_names="DATA/CHROMOPAINTER/{chrom}/{section}_{type}/donor_names.txt",
        haplo="DATA/CHROMOPAINTER/{chrom}/{section}_{type}/haplo.txt"
    log:
        "snakemake_logs/chromopainter_prep_{chrom}_{section}_{type}.log"
    params:
        n_sim = config.get("chromopainter_prep_n_sim", "100")
    run:
        datapreprocessing.prepare_chromopainter(input.data_ref, input.data_dec, "DATA/ETH/1KG_eth.txt", "DATA/CHROMOPAINTER/"+wildcards.chrom+"/"+wildcards.section+"_"+wildcards.type+"/",  params.n_sim)

rule chromopainter_recomb:
    input:
        proba_map="DATA/BED/genetic_map_sexavg.bdg",
        chromo_haplo="DATA/CHROMOPAINTER/{chrom}/{section}_vae/haplo.txt"
    output:
        "DATA/CHROMOPAINTER/{chrom}/recomb_{section}.txt"
    log:
        "snakemake_logs/chromopainter_recomb_{chrom}_{section}.log"
    shell:
        "Rscript SCRIPTS/recombfromhaplo.R DATA/CHROMOPAINTER/{wildcards.chrom}/ DATA/CHROMOPAINTER/{wildcards.chrom}/ {wildcards.section} {input.proba_map}"

rule paint_chromo:
    input:
        donor="DATA/CHROMOPAINTER/{chrom}/{section}_{type}/donor.txt",
        haplo="DATA/CHROMOPAINTER/{chrom}/{section}_{type}/haplo.txt",
        recomb="DATA/CHROMOPAINTER/{chrom}/recomb_{section}.txt"
    output:
        "DATA/CHROMOPAINTER/{chrom}/{section}_{type}/out_{section}_{type}.samples.out"
    log:
        "snakemake_logs/paint_chromo_{chrom}_{section}_{type}.log"
    shell:
        "SCRIPTS/chromopainter -J -g {input.haplo} -r {input.recomb} -f {input.donor} -o DATA/CHROMOPAINTER/{wildcards.chrom}/{wildcards.section}_vae/out_{wildcards.section}_vae"
