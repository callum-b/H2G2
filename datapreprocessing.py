### DATA PREPROCESSING

"""
  This is where data preprocessing functions are stored, such as converting .vcf files to haplotypic, building chromopainter input files, etc
"""


## imports
import numpy as np
import pandas as pd
import pyranges as pr
import gzip
import math
import csv
import re

def remove_worst_scores_until_no_overlap(gr):
    """
      remove_worst_scores_until_no_overlap: uses pyRanges to find and remove overlaps, keeping the best scores
      used in recombprobatohotspots
    """
    df = gr.df
    old_length = -1
    new_length = len(df)
    while old_length != new_length:
        df = df.drop(["Cluster", "Count"], axis=1, errors="ignore")
        df = pr.PyRanges(df).cluster(count=True).df.sort_values("Score", ascending=False)
        df = df[df.duplicated('Cluster', keep='last') | (df.Count == 1)]
        old_length = new_length
        new_length = len(df)
        
    return pr.PyRanges(df.drop("Cluster Count".split(), axis=1)).sort()


def recombproba_to_hotspots(probamap_f:str, chromosome:str, mindist=10000, thrs=1):
    """
      recombproba_to_hotspots: take a file describing the recombination probabilities over a whole chromosome (like the maps provided in Halldorsson et al., 2019),
        and keep the centres of high likelihood sections to create a list of hotspots
    """
    hotspots = pd.read_table(probamap_f, header=None)
    hotspots.columns = ["Chromosome", "Start", "End", "Score"]
    hotspots = hotspots.loc[hotspots.Chromosome == chromosome]
    hotspots = pr.PyRanges(hotspots[hotspots.Score>=thrs])
    
    hotspots.Start = ((hotspots.Start + hotspots.End)/2).astype(np.int64) - math.floor(mindist/2) # int is 64 bit
    hotspots.End = hotspots.Start + math.floor(mindist/2)+1
    hotspots
    clusters = remove_worst_scores_until_no_overlap(hotspots)
    clusters.Start = ((clusters.Start + clusters.End)/2).astype(np.int64)
    clusters.End = clusters.Start + 1
    
    return clusters


def hap_count(mycells:list) -> int:
    """
      hap_count: count the number of occurrences of a mutation in a line of a diploid vcf
      used in splitvcf
    """
    n=0
    for cell in mycells :
        if cell == "0|1" or cell == "1|0" :
            n+=1
        elif cell == "1|1" :
            n+=2
    return n


def split_vcf(delim_f:str, SNPtable_f:str, out_dir:str, maxn=5000, minn=500, thrs=1, keep_excl=False, sep_haplo=False) ->str:
    """
      splitvcf: take a gzipped input vcf file and split it into subsections based on a given list of delimiters, with params to define max and min numbers of mutations allowed per subsection, as well as a threshold for rare mutations
      does not load entire SNPtable in a dataframe because it can be VERY large
      returns completion message
    """
    if maxn<=minn:
        return "You've mixed up the max and min arguments, you pinstriped barbarian!"
    delims = list([int(x.split('\t')[1]) for x in open(delim_f).readlines()])
    delims.sort()
    
    with gzip.open(SNPtable_f, "rt") as csvfile:
        header = "#HEADER NOT FOUND :("
        data = []
        if keep_excl:
            excluded = open(out_dir + "excluded.vcf", "w")
        prev = 0
        delim = delims.pop(0) # get position of first hotspot
        csvread = csv.reader(csvfile, delimiter='\t',  quotechar='"')
        
        for myline in csvread: #reads csvfile line by line, storing array of cells into myline
            if re.search("^#", myline[0]) : #if current line is the header, store it
                header = myline
            else :
                if hap_count(myline[9:]) >= thrs :
                    data.append(myline)        
                elif keep_excl :
                    excluded.write("\t".join(myline)+"\n")
                if(int(myline[1])>delim) : #when we reach the hotspot, print all lines to a file
                    if len(data)<minn :
                        print("Small number of mutations in this section ("+str(len(data))+"), keeping for next output. Hotspot not used at "+str(delim))
                    else :
                        print("Hotspot reached at "+str(delim)+" (csv file line "+str(csvread.line_num)+"), writing "+str(len(data))+" lines to file...")
                        filepath = out_dir + str(prev)+"_"+str(delim)+".vcf"
                        f = open(filepath, 'a')
                        f.write("\t".join(header)+"\n")
                        for item in data:
                            f.write("\t".join(item)+"\n")
                        f.close()
                        if sep_haplo:
                            vcf_to_haplo(filepath, filepath.sep(".")[0:-1] + "_haplo.vcf")
                        print("Writing done. Moving on...")
                        data = []
                        prev = delim # Forget previous hotspot only if used for output
                    delim = delims.pop(0) # get pos of next hotspot
                if len(data)>=maxn :
                    print(str(maxn)+" mutations without a hotspot, writing to a new file!")
                    filepath = out_dir + str(prev)+"_"+str(myline[1])+".vcf"
                    f = open(filepath, 'a')
                    f.write("\t".join(header)+"\n")
                    for item in data:
                        f.write("\t".join(item)+"\n")
                    f.close()
                    if sep_haplo:
                        vcf_to_haplo(filepath, filepath.sep(".")[0:-1] + "_haplo.vcf")
                    prev = myline[1]
                    data = []
                    
    print("EOF reached, writing "+str(len(data))+" lines to file...\n")
    if len(data)<minn :
        print("Small number of mutations in this section ("+str(len(data))+"), appending to previous.")
        f = open(filepath, 'a')
        for item in data:
            f.write("\t".join(item)+"\n")
        f.close()
        if sep_haplo:
                vcf_to_haplo(filepath, filepath.sep(".")[0:-1] + "_haplo.vcf")
    else :
        filepath = out_dir + str(prev)+"_"+str(delim)+".vcf"
        f = open(filepath, 'a')
        f.write("\t".join(header)+"\n")
        for item in data:
            f.write("\t".join(item)+"\n")
        f.close()
        if sep_haplo:
                vcf_to_haplo(filepath, filepath.sep(".")[0:-1] + "_haplo.vcf")
        
    if keep_excl:
        excluded.close()
    return "Writing done. That's all, folks!\n"


def vcf_to_haplo(in_f:str, out_f:str, header=True, mode="w")->str:
    """
	  vcftohaplo: read vcf file as found in 1KG ftp, output haplotypic vcf file with condensed info
	  does not return new df object as it is FAR more efficient to run this once at the beginning of the project and then load new vcf with pd.read_csv() in each script
      returns completion message
    """
    ## colnames in file: #CHROM POS ID REF ALT QUAL FILTER INFO FORMAT HG...
    source = pd.read_csv(in_f, sep="\t", dtype="string")
    source["ID"] = source["#CHROM"] + "." + source["POS"] + source["REF"] + ">" + source["ALT"]
    out = open(out_f, mode)
    
    if header :
        newnames = []
        for n in source.columns[9:]:
            newnames.append(n+".A")
            newnames.append(n+".B")
        out.write("ID;" + ";".join(newnames) + "\n")
        
    for i in range(len(source)):
        out.write(source["ID"][i] + ";")
        templine = source.loc[i][9:]
        splitline = []
        for x in templine :
            splitline.extend(x.split("|"))
        out.write(";".join(splitline) + "\n")
        
    out.close()
    return "Done for "+in_f

def prepare_chromopainter(ref_f:str, sim_f:str, eth_f:str, outdir:str, nsim=100)->str:
    """
      preparechromopainter: combine ref and sim data to create chromopainter inputs (donor and haplo files)
      like vcf to haplo, creates files on disk and does not return any data
      returns completion message
    """
    refdata = pd.read_csv(ref_f, sep=";", header=0, index_col=0).transpose()
    simdata = pd.read_csv(sim_f, sep=";", header=0).transpose()
    simdata = simdata.round().astype(np.int32)
    if nsim > len(simdata):
        return "ERROR SIM DATA TOO SMALL PLEASE GENERATE MORE"
    ethgroups = {}
    n=0
    with open(eth_f, newline='') as ethfile:
        for myline in ethfile:
            ID, eth = myline.split()
            if eth not in ethgroups.keys():
                ethgroups[eth] = []
            if ID+".A" in refdata.index:
                ethgroups[eth].append(ID+".A")
                ethgroups[eth].append(ID+".B")
                n+=2

    donor_out = open(outdir + "donor.txt", "w")
    donor_names_out = open(outdir + "donor_names.txt", "w")
    haplo_out = open(outdir + "haplo.txt", "w")

    haplo_out.write(str(n) + "\n")
    haplo_out.write(str(n + nsim) + "\n")
    haplo_out.write(str(len(refdata.columns)) + "\n")
    haplo_out.write("P " + " ".join(re.sub(r'[A-z]+', '', x.split(".")[1].split(">")[0]) for x in refdata.columns) + "\n")
    haplo_out.write("".join(["S"] * len(refdata.columns)) + "\n")


    for myeth in sorted(ethgroups.keys()):
        donor_out.write(myeth + " " + str(len(ethgroups[myeth]))+"\n")
        donor_names_out.write(myeth + " " + str(len(ethgroups[myeth]))+" ")
        for myID in sorted(ethgroups[myeth]) :
            donor_names_out.write(myID +" ")
            haplo_out.write("".join(map(str, refdata.loc[myID])) + "\n")
        donor_names_out.write("\n")
            
    for i in range(nsim):
        haplo_out.write(''.join(map(str,list(simdata.iloc[i,])))+"\n")

    donor_out.close()
    donor_names_out.close()
    haplo_out.close()
    
    return "Done!"


