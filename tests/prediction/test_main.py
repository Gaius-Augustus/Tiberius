import tiberius, os, sys, pytest
import numpy as np
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
import tarfile
from pathlib import Path

def parse_gtf(gtf_path):
    """
    Parses a GTF file and returns a dictionary of transcripts and a set of CDS features.
    
    Returns:
      transcripts: A dict mapping transcript_id -> {"gene_id": gene_id, "cds": [ (chrom, start, end, strand), ... ]}
      cds_set: A set of tuples representing CDS features, e.g., (chrom, start, end, strand)
    """
    transcripts = {}
    cds_set = set()
    with open(gtf_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            fields = line.split("\t")
            if len(fields) < 9:
                continue
            chrom, source, feature, start, end, score, strand, phase, attributes = fields
            start, end = int(start), int(end)
            # We care only about CDS features for our CDS-level metric.
            if feature != "CDS":
                continue
            # Parse attributes to extract transcript_id and gene_id.
            attr_parts = [a.strip() for a in attributes.split(";") if a.strip()]
            attr_dict = {}
            for part in attr_parts:
                try:
                    key, val = part.split(" ", 1)
                    attr_dict[key] = val.strip('"')
                except ValueError:
                    continue
            transcript_id = attr_dict.get("transcript_id")
            gene_id = attr_dict.get("gene_id")
            if transcript_id is None:
                continue
            # Add the CDS to the transcript record.
            if transcript_id not in transcripts:
                transcripts[transcript_id] = {"gene_id": gene_id, "cds": []}
            cds_feat = (chrom, start, end, strand)
            transcripts[transcript_id]["cds"].append(cds_feat)
            cds_set.add(cds_feat)
    return transcripts, cds_set

def compute_f1(true_set, pred_set):
    """
    Computes the F1 score given two sets.
    """
    tp = len(true_set.intersection(pred_set))
    fp = len(pred_set - true_set)
    fn = len(true_set - pred_set)
    if tp == 0:
        return 0.0
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    return 2 * precision * recall / (precision + recall)

def get_transcript_set(transcripts):
    """
    Convert transcripts dictionary into a set for transcript-level comparison.
    Each transcript is represented as a frozenset of its CDS features.
    """
    transcript_set = set()
    for tid, data in transcripts.items():
        # Use frozenset so that the order of CDS features does not matter.
        transcript_set.add(frozenset(data["cds"]))
    return transcript_set
    
def test_assemble_transcript_forward() -> None:
    """
    Test that, for a gene on the forward strand, the assembled coding sequence is correctly 
    constructed and translated. In this example:
      - The genomic sequence is 'ATGGCCTAA', which is 9 bases long.
      - We define two exons: positions [1, 3] and [4, 9]. The joined sequence is
        "ATG" + "GCCTAA" = "ATGGCCTAA".
      - 'ATG' translates to 'M', 'GCC' translates to 'A', and 'TAA' is a stop codon,
        so the expected protein is "MA*".
    """
    seq_str = "ATGGCCTAA"
    record = SeqRecord(Seq(seq_str), id="chr1")
    exons = [[1, 3], [4, 9]]
    
    coding_seq, prot_seq = tiberius.assemble_transcript(exons, record, '+')
    
    assert coding_seq == Seq("ATGGCCTAA"), "Coding sequence is not as expected for forward strand."
    assert str(prot_seq) == "MA*", "Protein sequence is not as expected for forward strand."

def test_assemble_transcript_reverse() -> None:
    """
    Test that, for a gene on the reverse strand, exons are sorted in descending order and
    each exon sequence is reverse-complemented. In this test:
      - We use a genomic sequence 'TTAGGCCAT'.
      - Exons are given as [[1, 3], [4, 9]] in genomic coordinates.
      - For a negative strand gene, the function sorts exons in descending order.
        The exon covering positions [4, 9] is processed first:
            * Genomic substring: sequence[3:9] == 'GGCCAT'
            * Reverse complement of 'GGCCAT' is 'ATGGCC'
        Then the exon at [1, 3]:
            * Genomic substring: sequence[0:3] == 'TTA'
            * Reverse complement of 'TTA' is 'TAA'
      - The joined transcript is "ATGGCC" + "TAA" = "ATGGCCTAA", which translates to "MA*".
    """
    seq_str = "TTAGGCCAT"
    record = SeqRecord(Seq(seq_str), id="chr1")
    exons = [[1, 3], [4, 9]]
    
    coding_seq, prot_seq = tiberius.assemble_transcript(exons, record, '-')
    
    assert coding_seq == Seq("ATGGCCTAA"), "Coding sequence is not as expected for reverse strand."
    assert str(prot_seq) == "MA*", "Protein sequence is not as expected for reverse strand."

def test_assemble_transcript_invalid_length() -> None:
    """
    Test that if the assembled coding sequence length is valid (divisible by 3) but the 
    translation does NOT end with a stop codon, the function returns (None, None).
    
    Here, we use the sequence 'ATGGCC', which is 6 bases long and translates to 'MA' (no stop).
    """
    seq_str = "ATGGCC"
    record = SeqRecord(Seq(seq_str), id="chr1")
    exons = [[1, 6]]
    
    coding_seq, prot_seq = tiberius.assemble_transcript(exons, record, '+')
    
    assert coding_seq is None and prot_seq is None, "Function should return (None, None) when no stop codon is present."


def test_group_sequences_basic() -> None:
    """
    Test grouping with a simple dataset using custom parameters.
    
    Here, we use t=10 and chunk_size=5 to force grouping.
    Given unsorted input:
      - seq_names: ["A", "B", "C", "D"]
      - seq_lens:  [1, 2, 7, 8]
      
    Processing (the function sorts by length in ascending order):
      Sorted pairs: [("A", 1), ("B", 2), ("C", 7), ("D", 8)]
      
      For ("A", 1): 1 < 5 so add 5. current_sum becomes 5. current_group = ["A"]
      For ("B", 2): 2 < 5 so add 5. current_sum becomes 10. current_group = ["A", "B"]
      For ("C", 7): 7 >= 5 so add 7. current_sum becomes 17. current_group = ["A", "B", "C"]
          -> 17 > 10 triggers grouping: groups += [["A", "B", "C"]]; then reset current_group and current_sum.
      For ("D", 8): 8 >= 5 so add 8. current_sum becomes 8; current_group = ["D"]
      
      At the end, the remaining current_group is appended.
      
    Expected groups: [["A", "B", "C"], ["D"]]
    """
    seq_names = ["A", "B", "C", "D"]
    seq_lens = [1, 2, 7, 8]
    groups = tiberius.group_sequences(seq_names, seq_lens, t=10, chunk_size=5)
    assert groups == [["A", "B", "C"], ["D"]]

def test_group_sequences_sorted_order() -> None:
    """
    Test that group_sequences sorts the input sequences by length before grouping.
    
    Given:
      - seq_names: ["X", "Y", "Z"] with corresponding lengths [8, 3, 5] (unsorted order)
      
    After sorting by increasing length, the order should be:
      [("Y", 3), ("Z", 5), ("X", 8)]
    
    With t=10 and chunk_size=5:
      For ("Y",3): 3 < 5 so add 5. current_sum = 5; group = ["Y"]
      For ("Z",5): 5 is not less than 5, so add 5 (its actual length). current_sum = 10; group = ["Y", "Z"]
      For ("X",8): 8 >= 5 so add 8. current_sum becomes 18; group = ["Y", "Z", "X"]
         -> 18 > 10 triggers grouping.
         
    Expected output: [["Y", "Z", "X"]]
    """
    seq_names = ["X", "Y", "Z"]
    seq_lens = [8, 3, 5]
    groups = tiberius.group_sequences(seq_names, seq_lens, t=10, chunk_size=5)
    assert groups == [["Y", "Z", "X"]]

def test_group_sequences_no_grouping() -> None:
    """
    Test that if the total contribution never exceeds the threshold t,
    all sequences are returned in a single group.
    
    With t=1000 and chunk_size=5, using two sequences with lengths less than chunk_size:
      - Each sequence adds 5.
      - Total current_sum = 5 + 5 = 10, which is less than 1000.
      
    Expected output: [["A", "B"]]
    """
    seq_names = ["A", "B"]
    seq_lens = [1, 2]  # Both values are less than chunk_size (5), so each adds 5.
    groups = tiberius.group_sequences(seq_names, seq_lens, t=1000, chunk_size=5)
    assert groups == [["A", "B"]]


@pytest.mark.integration
def test_full_workflow(tmp_path, monkeypatch) -> None:
    """
    Integration test that exercises the entire workflow:
      - Downloads the model from the given URL.
      - Reads a longer genome FASTA file.
      - Runs predictions.
      - Writes the output GTF file.
      
    This test actually accesses external resources, so it is marked as an integration test.
    """
     # Locate the test data tar.gz file relative to this test script.
    current_dir = Path(__file__).parent
    tar_file = current_dir / ".." / ".." / "test_data" / "Panthera_pardus" / "inp.tar.gz"
    assert tar_file.exists(), f"Test data tar.gz file not found: {tar_file}"
    
    # Create a temporary extraction directory.
    extract_dir = tmp_path / "extracted_genome"
    extract_dir.mkdir()
    
    # Extract the tar.gz file.
    with tarfile.open(tar_file, "r:gz") as tar:
        tar.extractall(path=extract_dir)
    
    # The expected result is that genome.fa is present after extraction.
    genome_file = extract_dir / "inp" / "genome.fa"
    assert genome_file.exists(), "Extracted genome.fa file not found."
    
    # Define the output GTF file path in tmp_path.
    output_gtf = tmp_path / "output.gtf"
    

    test_args = [
        "main.py",
        "--genome", str(genome_file),
        "--out", str(output_gtf),
        "--seq_len", "500004",
        "--batch_size", "8",
        "--strand", "+",
        "--id_prefix", "integration_",
       # "--model_lstm", "model_weights/tiberius_weights_lstm/"
    ]
    
    # Monkeypatch sys.argv so that main() sees our test arguments.
    monkeypatch.setattr(sys, "argv", test_args)
    
    # Run the main workflow.
    try:
        tiberius.main()
    except SystemExit as e:
        pytest.fail(f"main() exited unexpectedly with exit code {e.code}")
    
    # Verify that the output GTF file was created.
    assert output_gtf.exists(), "The output GTF file was not created."
    
    ref_gtf = extract_dir / "inp" / "annot_+.gtf"
    assert ref_gtf.exists(), f"Reference GTF not found: {ref_gtf}"
    
    ref_transcripts, ref_cds = parse_gtf(str(ref_gtf))
    pred_transcripts, pred_cds = parse_gtf(str(output_gtf))
    
    cds_f1 = compute_f1(ref_cds, pred_cds)
    
    ref_transcript_set = get_transcript_set(ref_transcripts)
    pred_transcript_set = get_transcript_set(pred_transcripts)
    transcript_f1 = compute_f1(ref_transcript_set, pred_transcript_set)
    
    # Print computed F1 scores for debugging information.
    print(f"CDS F1 Score: {cds_f1:.3f}")
    print(f"Transcript F1 Score: {transcript_f1:.3f}")
    
    assert cds_f1 >= 0.75, f"CDS-level F1 score too low: {cds_f1:.3f}"
    assert transcript_f1 >= 0.35, f"Transcript-level F1 score too low: {transcript_f1:.3f}"

