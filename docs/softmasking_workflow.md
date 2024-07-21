
### Recommended Workflow for Softmasking an Input Genome for Tiberius

This workflow was adapted by Katharina Hoff from EukSpecies-BRAKER2, available at https://github.com/gatech-genemark/EukSpecies-BRAKER2.

1. Use RepeatModeler2 (version v2.0.2) to build a species-specific database (`$DB`) from each species' genome in FASTA format (`${GENOME}`):
```shell
BuildDatabase -name ${DB} ${GENOME}
RepeatModeler -database ${DB} -pa 72 -LTRStruct
```

2. Use RepeatMasker v4.1.4 with that repeat library to initially mask the genomes, resulting in a masked FASTA file `$SPECIES.fa.masked`:

```shell
RepeatMasker -pa 72 -lib ${DB} -xsmall ${SPECIES}.fa
```

3. Sometimes the previous steps miss some repetitive regions. Therefore, run Tandem Repeats Finder as an additional masking step:

```shell
mkdir trf
cd trf
ln -s ../genome.fa.masked genome.fa
splitMfasta.pl --minsize=25000000 genome.fa
ls genome.split.*.fa | parallel 'trf {} 2 7 7 80 10 50 500 -d -m -h &> {}.log'
ls genome.split.*.fa.2.7.7.80.10.50.500.dat | parallel 'parseTrfOutput.py {} --minCopies 1 \
    --statistics {}.STATS > {}.raw.gff 2> {}.parsedLog'
ls genome.split.*.fa.2.7.7.80.10.50.500.dat.raw.gff | parallel 'sort -k1,1 -k4,4n -k5,5n {} \ 
    > {}.sorted 2> {}.sortLog'
FILES=genome.split.*.fa.2.7.7.80.10.50.500.dat.raw.gff.sorted
for f in $FILES do
    bedtools merge -i $f | awk 'BEGIN{OFS="\t"} {print $1,"trf","repeat",$2+1,$3,".",".",".","."}' \
        > $f.merged.gff 2> $f.bedtools_merge.log
done
ls genome.split.*.fa | parallel 'bedtools maskfasta -fi {} -bed \
    {}.2.7.7.80.10.50.500.dat.raw.gff.sorted.merged.gff -fo {}.combined.masked -soft &> {}.bedools_mask.log'
cat genome.split.*.fa.combined.masked > genome.fa.combined.masked
```


### References
* Flynn, Jullien M., et al. "RepeatModeler2 for automated genomic discovery of transposable element families." Proceedings of the National Academy of Sciences 117.17 (2020): 9451-9457.
* Benson, Gary. "Tandem repeats finder: a program to analyze DNA sequences." Nucleic acids research 27.2 (1999): 573-580.

