SNPpath="../../data/PRS/Field22418/merged_chromosome"
rawpath="../../data/PRS/ChronicGWAS/"
filepath=$rawpath"process/"
snppath="../../data/PRS/hg38.dbSNP150.bed"
targetpath="../../results/PRS/"

Rscript ChronicGWAS_step1.R \
  --path=$rawpath

ls $filepath*_* | xargs -n 1 basename > $filepath"filelist"
while read line 
do
file=$(echo $line | rev | cut -d'_' -f2- | rev)
if [ -e $targetpath$file".P.sscore" ]
then
echo $file" Finished!"
else
if echo $line | grep -q "ready"; then
    awk '!seen[$1]++' $filepath$line > removedump4
    awk '{print $1,$5}' removedump4 > SNP.pvalue4
    
plink2 \
  --bfile $SNPpath \
  --score removedump4 1 2 4 \
  --q-score-range Pvalue SNP.pvalue4 min \
  --out $targetpath$file
rm removedump4
rm SNP.pvalue4
fi

if echo $line | grep -q "raw"; then
    sortbedpath=$filepath"sort$file.bed"
    tempbedpath=$filepath"temp.bed"
    bedpath=$filepath"$file.bed"
    splitpath=$filepath"split$file.bed"
    outpath=$filepath$file
    sort -k1,1 -k2,2V $filepath$line > $sortbedpath
    awk -vOFS="\t" '{ printf "%s\t%d\t%d\n", $1, ($2 - 1), $2; }' $filepath$line | sort-bed - > $tempbedpath
    bedmap --echo --echo-map-id-uniq --delim '\t' $tempbedpath $snppath > $bedpath
    awk -F'\t' '{split($4, arr, ";"); print $1, $2, $3, arr[1]}' $bedpath > $splitpath
    Rscript ChronicGWAS_step2.R \
      --data=$sortbedpath \
      --map=$splitpath \
      --out=$outpath
    awk '!seen[$4]++' $outpath > removedump4
    awk '{print $4,$3}' removedump4 > SNP.pvalue4
plink2 \
  --bfile $SNPpath \
  --score removedump4 4 1 2 \
  --q-score-range Pvalue SNP.pvalue4 min \
  --out $targetpath$file
rm removedump4
rm SNP.pvalue4
rm $sortbedpath $tempbedpath $outpath $bedpath $splitpath
fi
fi
done < $filepath"filelist"
rm $filepath/filelist