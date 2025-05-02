import tiberius, os, sys, pytest
import tarfile
import numpy as np
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
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
        "--seq_len", "259992",
        "--batch_size", "8",
        "--strand", "+",
        "--id_prefix", "integration_"
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
    assert transcript_f1 >= 0.28, f"Transcript-level F1 score too low: {transcript_f1:.3f}"

