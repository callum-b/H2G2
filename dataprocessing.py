### DATA PROCESSING

## This is where data preprocessing functions are stored, such as converting .vcf files to haplotypic, building chromopainter input files, etc


## imports
import numpy as np
import pandas as pd
import pyranges as pr
import math
import csv
import re

def remove_worst_scores_until_no_overlap(gr):
    ## remove_worst_scores_until_no_overlap: uses pyRanges to find and remove overlaps, keeping the best scores
	## used in recombprobatohotspots
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


def recombprobatohotspots(probamap_f, chromosome, mindist=10000, thrs=1):
    ## recombprobatohotspots: take a file describing the recombination probabilities over a whole chromosome (like the maps provided in Halldorsson et al., 2019) and keep the centres of high likelihood sections to create a list of hotspots
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


def hap_count(mycells):
    ## hap_count: count the number of occurrences of a mutation in a line of a diploid vcf
    ## used in splitvcf
    n=0
    for cell in mycells :
        if cell == "0|1" or cell == "1|0" :
            n+=1
        if cell == "1|1" :
            n+=2
    return n


def splitvcf(delim_f, SNPtable_f, out_dir, maxn=5000, minn=500, thrs=1, keep_excl=False):
    ## splitvcf: take an input vcf file and split it into subsections based on a given list of delimiters, with params to define max and min numbers of mutations allowed per subsection, as well as a threshold for rare mutations
    ## does not load entire SNPtable in a dataframe because it can be VERY large
    if maxn<=minn:
        return "You've mixed up the max and min arguments, you pinstriped barbarian!"
    delims = list([int(x.split('\t')[1]) for x in open(delim_f).readlines()])
    delims.sort()
    
    with open(SNPtable_f, newline='') as csvfile:
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
                        f.close
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
                    f.close
                    prev = myline[1]
                    data = []
                    
    print("EOF reached, writing "+str(len(data))+" lines to file...\n")
    if len(data)<minn :
        print("Small number of mutations in this section ("+str(len(data))+"), appending to previous.")
        f = open(filepath, 'a')
        for item in data:
            f.write("\t".join(item)+"\n")
        f.close
    else :
        filepath = out_dir + str(prev)+"_"+str(delim)+".vcf"
        f = open(filepath, 'a')
        f.write("\t".join(header)+"\n")
        for item in data:
            f.write("\t".join(item)+"\n")
        f.close
        
    if keep_excl:
        excluded.close
    return "Writing done. That's all, folks!\n"


def vcftohaplo(in_f, out_f, header=True, mode="w"):
	## vcftohaplo: read vcf file as found in 1KG ftp, output haplotypic vcf file with condensed info
	## does not return new df object as it is FAR more efficient to run this once at the beginning of the project and then load new vcf with pd.read_csv() in each script
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
        
    out.close
    return "Done for "+in_f

