import argparse
import sys, os
import numpy as np
from tiberius import DataGenerator

decode = np.array(['a', 'c', 'g', 't', 'N',])

def get_sequences_from_example(x,y):
    for k in range(x.shape[0]):
        x_seq = np.array(x[k,:,:5]).argmax(axis=-1)
        y_seq = np.array(y[k]).argmax(axis=-1)
        # find tuples within the sequence y where it starts with 7 (start codon) and ends with 14 (stop codon)
        starts = np.where(y_seq==7)[0]
        stops = np.where(y_seq==14)[0]
        sequences = []
        for start in starts:
            for stop in stops:
                if stop > start + 3 :
                    # get decoded section from x_seq
                    seq = ''.join(decode[x_seq[start:stop+1]])
                    if seq.startswith("atg"):
                        sequences.append(seq)
                    elif seq.endswith("cat"):
                        # append reverse complement
                        rev_comp = seq[::-1].translate(str.maketrans("acgt", "tgca"))
                        sequences.append(rev_comp)
                    break
    
    return sequences


def main():
    parser = argparse.ArgumentParser(
        description="Load a TensorFlow model, remove the last layer, and save the modified model."
    )
    parser.add_argument(
        "--tfrecords_dir", type=str, required=True,
        help="Path to the directory of the tfrecords."
    )
    parser.add_argument(
        "--tfrecords_prefix", type=str, required=True,
        help="Prefix of the tfrecord files to read (e.g., species name)."
    )
    parser.add_argument(
        "--output_fasta", type=str, required=True,
        help="Path to save the output fasta file with the extracted sequences."
    )
    parser.add_argument(
        "--num_records", type=int, default=10,
        help="Number of records to process from the tfrecords."
    )
    parser.add_argument(
        "--number_sequences", type=int, default=50,
        help="Number of sequences to extract."
    )
    parser.add_argument(
        "--ref_annot", type=str, required=False,
        help="Reference annotation in GTF format to compute reference statistics."
    )
    parser.add_argument(
        "--genome", type=str, required=False,
        help="Genome in FASTA format to compute reference statistics."
    )
    parser.add_argument(
        "--min_seq_length", type=int, default=500000,
        help="Minimum sequence length to consider for reference statistics."
    )
    
    args = parser.parse_args()

    # create list of tfrecords files
    tfrecords_files = [f"{args.tfrecords_dir}/{args.tfrecords_prefix}_{i}.tfrecords" for i in range(args.num_records) if \
            os.path.exists(f"{args.tfrecords_dir}/{args.tfrecords_prefix}_{i}.tfrecords")]
    print(f"Found {len(tfrecords_files)} tfrecord files.")
    
    if len(tfrecords_files) == 0:
        sys.exit(1)

    # create data generator
    generator = DataGenerator(
        file_path=tfrecords_files, 
        batch_size=1, 
        shuffle=False,
        repeat=False,
        filter=False,
        output_size=15,
        hmm_factor=0,
        seq_weights=False, 
        softmasking=True,
        clamsa=False ,
        oracle=False,
        threads=5,
        tx_filter=[],
      )

    dataset1 = generator.get_dataset()

    sequences = []
    x_count = np.zeros((5), int)
    y_count = np.zeros((15), int)
    for x, y, _ in dataset1:
        chunk_size = x.shape[1]
        if len(sequences) < args.number_sequences:
            sequences += get_sequences_from_example(x,y)
        x_count += np.sum(np.array(x[:,:,:5]), axis=(0,1))
        y_count += np.sum(np.array(y), axis=(0,1))

    print("Nucleotide counts in TfRecords  (+ and - strand):")
    for i, nuc in enumerate(['A', 'C', 'G', 'T', 'N']):
        print(f"{nuc}: {x_count[i]} ({x_count[i]/np.sum(x_count[:5])*100:.2f}%)")

    if args.genome:
        # compute nucleotide frequencies in genome for all sequences longer than min_seq_length
        # and that fit into chunks of chunk_size without padding
        from Bio import SeqIO
        genome_counts = np.zeros((5), int)
        for record in SeqIO.parse(args.genome, "fasta"):
            seq = str(record.seq).lower()
            if len(seq) >= args.min_seq_length:
                trim_len = len(seq) - (len(seq) % chunk_size)
                seq = seq[:trim_len]
                a_count = seq.count('a')
                c_count = seq.count('c')
                g_count = seq.count('g')
                t_count = seq.count('t')
                n_count = seq.count('n') + seq.count('x')
                genome_counts += np.array([a_count, c_count, g_count, t_count, n_count])
        print("\nNucleotide counts in reference genome (only + strand):")
        for i, nuc in enumerate(['A', 'C', 'G', 'T', 'N']):
            print(f"{nuc}: {genome_counts[i]} ({genome_counts[i]/np.sum(genome_counts[:5])*100:.2f}%)")

        print("\nNucleotide counts in reference genome (+ and - strand):")
        print(f"A/T:{genome_counts[0] + genome_counts[3]} ({(genome_counts[0] + genome_counts[3])/(np.sum(genome_counts[:5])*2)*100:.2f}%)")
        print(f"C/G:{genome_counts[1] + genome_counts[2]} ({(genome_counts[1] + genome_counts[2])/(np.sum(genome_counts[:5])*2)*100:.2f}%)")
        print(f"N: {genome_counts[4]*2} ({genome_counts[4]/np.sum(genome_counts[:5])*100:.2f}%)")

    print("\nLabel counts in TfRecords:")
    labels = ["Intergenic",
            "I0", "I1", "I2",
            "E0","E1", "E2",
            "START",
            "EI0", "EI1", "EI2",
            "IE0", "IE1", "IE2",
            "STOP",]
    for i, label in enumerate(labels):
        print(f"{label}: {y_count[i]} ({y_count[i]/np.sum(y_count)*100:.2f}%)")
    
    if args.ref_annot:
        # compute label frequencies in reference annotation
        with open(args.ref_annot, "r") as f:
            gtf_lines = f.readlines()

        numb_exons = 0
        numb_introns = 0
        total_exon_length = 0
        total_intron_length = 0
        numb_transcripts = 0
        transcript_ids = []
        for line in gtf_lines:
            if line.startswith("#"):
                continue
            fields = line.strip().split("\t")
            if len(fields) < 9:
                continue
            feature = fields[2]
            start = int(fields[3])
            end = int(fields[4])
            if "transcript_id " in fields[-1]:
                transcript_ids.append(fields[-1].split('transcript_id "')[1].split('"')[0])
            length = end - start + 1
            if feature == "CDS":
                numb_exons += 1
                total_exon_length += length
            elif feature == "intron":
                numb_introns += 1
                total_intron_length += length    
        numb_transcripts = len(set(transcript_ids))


        print("\nReference annotation statistics:")
        print(f"Number of transcripts: {numb_transcripts}, (approx. number of START: {y_count[7]} and STOP: {y_count[14]})")
        print(f"Number of introns: {numb_introns}, (approx. sums of EI0 + EI1 + EI2: {np.sum(y_count[8:11])} and IE0 + IE1 + IE2: {np.sum(y_count[11:14])})")
        print(f"Total CDS length: {total_exon_length}, (approx. sum of E0 + E1 + E2 + 2*numb_exons: {np.sum(y_count[4:7]) + 2*numb_exons} )")
        print(f"Total inton length: {total_intron_length}, (approx. sum of I0 + I1 + I2: {np.sum(y_count[1:4])})")

    # write sequences to fasta file
    with open(args.output_fasta, "w") as f:
        for i, seq in enumerate(sequences[:args.number_sequences]):
            f.write(f">seq_{i}\n")
            f.write(f"{seq}\n")
    

if __name__ == "__main__":
    main()
