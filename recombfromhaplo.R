library(readr)
library(GenomicRanges)

options(scipen=999) # Otherwise will print in scientific notation

args = commandArgs(trailingOnly=TRUE)
print(args)

if (length(args)==0) {
	stop("Please provide working directory, section for analysis and chromosomal map.n", call.=FALSE)
} else {
    work_dir = args[1]
	section = args[2]
    map_file = args[3]
}

in_path = paste(work_dir, "haplo_", section, ".txt", sep="")
out_path = paste(work_dir, "recomb_", section, ".txt", sep="")

# read positions of mutations in section for study as GRanges object
# score is recomb rate expressed in Morgan per bp, will be calculated later in script
positions = readLines(in_path,n=4)
positions = positions[length(positions)]
positions = unlist(strsplit(positions, " "))
positions = strtoi(positions[2:length(positions)])
endpos = positions[2:length(positions)] -1
startpos = positions[1:length(positions)-1]
SNP_distances = data.frame(rep("chr1"), startpos, endpos, rep(-1))
colnames(SNP_distances) = c("chr", "start", "end", "score")

SNP_distances_gr = makeGRangesFromDataFrame(SNP_distances, TRUE, TRUE)

# read distances from chromosomal map as GRanges
distances = read.csv(map_file, header=1, sep=" ")
colnames(distances) = c("start", "cMperMb", "cMatstart")
endpos = distances$start 
endpos = c(endpos[2:length(endpos)], endpos[length(endpos)])
distances$end = endpos -1
distances$chr = rep("chr1", length(endpos))
distances = distances[,c(5,1,4,2,3)]

distances_gr = makeGRangesFromDataFrame(distances, TRUE, TRUE)


# use built-in GRanges overlap function to find distances for mutations with constant distance between them
simplematches = findOverlaps(SNP_distances_gr, distances_gr, type="within")
for (n in 1:length(simplematches)){
	SNP_distances$score[queryHits(simplematches)[n]] = distances_gr$cMperMb[subjectHits(simplematches)[n]] / 10000 ## Mb --> b and cM --> M
}

# iterate over GRanges overlap object to resolve mutations for which weighted average distances must be calculated
complex_tomatch = makeGRangesFromDataFrame(SNP_distances[SNP_distances$score == -1,], TRUE, TRUE)

for (n in 1:length(complex_tomatch)){
	myquery = complex_tomatch[n]
	print(names(myquery))
	myoverlaps = findOverlaps(myquery, distances_gr)
	print(length(myoverlaps))
	myscore=0
	for (n in 1:length(myoverlaps)){
		myol = myoverlaps[n]
		mytarget = distances_gr[subjectHits(myol)]
		mystartpos = max(start(myquery), start(mytarget))
		myendpos = min(end(myquery), end(mytarget))
		myscore = myscore + (myendpos - mystartpos) / 10000 * mytarget$cMperMb 
	}
	SNP_distances$score[strtoi(names(myquery))] = myscore / width(myquery)
}

write.table(SNP_distances[c(2,4)], file=out_path, quote=FALSE, sep=" ", col.names=c("start.pos", "recom.rate.perbp"), row.names=FALSE)
write(paste(positions[length(positions)], " 0"), file=out_path, append=TRUE)

print("Done!")
