import tiberius, os, sys, pytest
import numpy as np
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from pathlib import Path
    
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

