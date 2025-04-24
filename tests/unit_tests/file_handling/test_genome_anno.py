import os
import sys
import csv
import gzip
import bz2
import tempfile

import pytest

from tiberius.genome_anno import (
    NotGtfFormat,
    Transcript,
    Anno
)

def make_gtf_line(chr_, src, feature, start, end, strand, phase, attrs):
    """Helper to build a GTF line list."""
    return [
        chr_, src, feature,
        start, end,
        '.', strand, str(phase), attrs
    ]

class TestTranscript:

    @pytest.fixture
    def tx(self):
        # start with empty gene_id so add_line will set gene_id from attrs
        return Transcript(id="tx1", gene_id="", chr="chr1", source_anno="srcA", strand="+")

    def test_add_line_basic_and_gene_id(self, tx):
        line = make_gtf_line(
            "chr1", "src1", "CDS", 10, 20, "+", 0,
            'gene_id "gX"; transcript_id "tx1";'
        )
        tx.add_line(line.copy())
        # should have one CDS entry
        assert "CDS" in tx.transcript_lines
        assert tx.transcript_lines["CDS"][0][3:] == [10, 20, '.', '+', '0', 'gene_id "gX"; transcript_id "tx1";']
        # start and end updated
        assert tx.start == 10
        assert tx.end == 20
        # gene_id pulled from attributes
        assert tx.gene_id == "gX"
        # source_method updated
        assert tx.source_method == "src1"

    def test_add_line_wrong_chr_and_strand_raises(self, tx):
        bad = make_gtf_line("chr2", "src", "exon", 1, 5, "-", 0, "")
        # chr mismatch and strand mismatch → both false in (line[0]==chr or line[6]==strand)
        with pytest.raises(NotGtfFormat):
            tx.add_line(bad)

    def test_get_type_coords_and_length(self, tx):
        # add mixed CDS phases and an exon
        lines = [
            make_gtf_line("chr1","s","CDS",5,10,"+",1,'gene_id "g"; transcript_id "tx1";'),
            make_gtf_line("chr1","s","CDS",15,18,"+",2,'gene_id "g"; transcript_id "tx1";'),
            make_gtf_line("chr1","s","exon",25,30,"+",0,'gene_id "g"; transcript_id "tx1";')
        ]
        for L in lines:
            tx.add_line(L.copy())
        # frame=True
        coords = tx.get_type_coords("CDS", frame=True)
        # should have keys '0','1','2'
        assert set(coords.keys()) == {"0","1","2"}
        # phase '1' contains [5,10]
        assert coords["1"] == [[5,10]]
        # phase '2' contains [15,18]
        assert coords["2"] == [[15,18]]
        # frame=False
        flat = tx.get_type_coords("CDS", frame=False)
        assert flat == [[5,10],[15,18]]
        # get_cds_len sums inclusive lengths
        assert tx.get_cds_len() == (10-5+1) + (18-15+1)

    def test_add_missing_lines_includes_introns_and_codons(self):
        # Build a tx with two CDS segments
        tx = Transcript("tx2","g2","chr1","src","-")
        cds1 = make_gtf_line("chr1","s","CDS",10,12,"-",0,'gene_id "g2"; transcript_id "tx2";')
        cds2 = make_gtf_line("chr1","s","CDS",20,22,"-",0,'gene_id "g2"; transcript_id "tx2";')
        tx.add_line(cds1.copy())
        tx.add_line(cds2.copy())
        # no intron yet
        assert "intron" not in tx.transcript_lines
        ok = tx.add_missing_lines()
        assert ok is True
        # now has intron, transcript, start_codon, stop_codon
        for k in ("intron","transcript","start_codon","stop_codon"):
            assert k in tx.transcript_lines

    def test_add_missing_lines_no_cds_exon_returns_false(self):
        tx = Transcript("tx3","","chr1","src","+")
        # add only a gene-level line
        gene = make_gtf_line("chr1","s","gene",1,5,"+","0",'gene_id "g3";')
        tx.transcript_lines = {"gene":[gene.copy()]}
        tx.start = tx.end = -1
        result = tx.add_missing_lines()
        assert result is False

    def test_redo_phase_and_check_splits(self):
        tx = Transcript("tx4","g4","chr1","src","+")
        # Two small CDS segments adjacent
        cds1 = make_gtf_line("chr1","s","CDS",1,3,"+",0,'')
        cds2 = make_gtf_line("chr1","s","CDS",4,6,"+",2,'')
        tx.add_line(cds1.copy())
        tx.add_line(cds2.copy())
        # duplicate a split segment to test merge
        tx.transcript_lines["CDS"].append(make_gtf_line("chr1","s","CDS",7,9,"+",1,''))
        # Now phases are as given; redo them
        print(tx.transcript_lines)
        tx.redo_phase()
        # Check phases recomputed: first exon phase=0, second gets new
        phs = [line[7] for line in tx.transcript_lines["CDS"]]
        assert phs[0] == 0
        # now test check_splits merges adjacent [1-3] and [4-6]
        tx.transcript_lines["CDS"] = [
            make_gtf_line("chr1","s","CDS",1,3,"+",0,''),
            make_gtf_line("chr1","s","CDS",4,6,"+",0,''),
        ]
        tx.check_splits()
        # should merge to a single region [1,6]
        assert len(tx.transcript_lines["CDS"]) == 1
        assert tx.transcript_lines["CDS"][0][3:5] == [1,6]

    def test_get_gtf_sorts_and_prefixes(self):
        tx = Transcript("tx5","g5","chr2","srcB","+")
        # add an exon and CDS
        exon = make_gtf_line("chr2","s","exon",5,8,"+",0,'')
        cds  = make_gtf_line("chr2","s","CDS",6,7,"+",0,'')
        tx.add_line(exon.copy())
        tx.add_line(cds.copy())
        tx.add_missing_lines()
        out = tx.get_gtf(prefix="PFX")
        # first line must be the transcript line with prefix
        assert out[0][2] == "transcript"
        assert out[0][8] == "PFX.tx5"
        # CDS lines contain cds_type in attribute and prefix in transcript_id
        cds_lines = [l for l in out if l[2]=="CDS"]
        assert any("cds_type=" in l[8] and "PFX.tx5" in l[8] for l in cds_lines)
        # exon lines added if missing
        assert any(l[2]=="exon" for l in out)

class TestAnno:

    @pytest.fixture
    def simple_gtf(self, tmp_path):
        p = tmp_path / "anno.gtf"
        lines = [
            # gene line
            ["chr1","A","gene","1","100",".","+",".","g1"],
            # transcript line
            ["chr1","A","transcript","10","90",".","+",".","tx1"],
            # CDS lines
            ["chr1","A","CDS","20","30",".","+","0",'transcript_id "tx1"; gene_id "g1";'],
            ["chr1","A","CDS","40","50",".","+","1",'transcript_id "tx1"; gene_id "g1";'],
        ]
        with open(p,"w") as f:
            w=csv.writer(f,delimiter="\t")
            w.writerows(lines)
        return str(p)

    def test_addGtf_and_norm_and_find_genes_and_get_gtf(self, simple_gtf):
        anno = Anno(path=simple_gtf, id="annoX")
        anno.addGtf()
        # Should have transcript tx1
        assert "tx1" in anno.transcripts
        # Norm: nothing to remove since CDS present
        anno.norm_tx_format()
        # find_genes builds gene_gtf and genes
        anno.find_genes()
        # gene g1 should map to [tx1]
        assert anno.genes["g1"] == ["tx1"]
        # get_gtf returns a list with gene line first, then transcript+CDS etc.
        out = anno.get_gtf()
        assert out[0][2] == "gene"
        assert any(l[2]=="transcript" for l in out)
        assert any(l[2]=="CDS" for l in out)

    def test_get_subset_and_change_id_and_list(self, simple_gtf):
        anno = Anno(path=simple_gtf, id="annoY")
        anno.addGtf()
        # subset of tx1
        subset = anno.get_subset(["tx1"])
        assert set(subset.keys()) == {"tx1"}
        # change annotation ID
        anno.change_id("newID")
        assert anno.id == "newID"
        # all transcripts.Source_anno updated
        for tx in anno.transcripts.values():
            assert tx.source_anno == "newID"
        # get_transcript_list returns Transcript instances
        lst = anno.get_transcript_list()
        assert all(isinstance(t, Transcript) for t in lst)

    def test_rename_tx_ids_and_write(self, simple_gtf, tmp_path):
        anno = Anno(path=simple_gtf, id="annoZ")
        anno.addGtf()
        anno.norm_tx_format()
        anno.find_genes()
        # rename with prefix
        tab = anno.rename_tx_ids(prefix="P")
        # translation_tab entries like [new_tx, old_tx]
        assert all(old.startswith("tx") for new, old in tab)
        # gene_gtf keys updated to "Pg1", "Pg2",...
        assert any(g.startswith("P_g") or g.startswith("Pg") for g in anno.genes)
        # write out and read back
        outp = tmp_path / "out.gtf"
        anno.write_anno(str(outp))
        with open(outp) as f:
            lines = [l.strip().split("\t") for l in f]
        # should match get_gtf
        expected = anno.get_gtf()
        expected = [list(map(str, e)) for e in expected]
        assert lines == expected
