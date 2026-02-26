from ._version import __version__

__all__ = [
    "__version__",
    # Optional: export “public” symbols too
    "GenePredHMMLayer",
    "GenePredHMMEmitter",
    "GenePredHMMTransitioner",
    "GenePredMultiHMMTransitioner",
    "SimpleGenePredHMMTransitioner",
    "GenomeSequences",
    "GeneStructure",
    "Annotation",
    "Anno",
    "DataGenerator",
    "PredictionGTF",
    "parseCmd",
    "assemble_transcript",
    "group_sequences",
    "main",
    "run_tiberius",
]

def __getattr__(name: str):


    if name == "GenomeSequences":
        from .genome_fasta import GenomeSequences
        return GenomeSequences

    if name in {"GeneStructure", "Annotation"}:
        from .annotation_gtf import GeneStructure, Annotation
        return locals()[name]

    if name == "Anno":
        from .genome_anno import Anno
        return Anno

    if name in {"custom_cce_f1_loss", "lstm_model", "Cast"}:
        from .models import custom_cce_f1_loss, lstm_model, Cast
        return locals()[name]

    if name == "DataGenerator":
        from .data_generator import DataGenerator
        return DataGenerator

    if name == "PredictionGTF":
        from .eval_model_class import PredictionGTF
        return PredictionGTF

    if name == "parseCmd":
        from .tiberius_args import parseCmd
        return parseCmd

    if name in {"assemble_transcript", "group_sequences", "main", "run_tiberius"}:
        from .main import assemble_transcript, group_sequences, main, run_tiberius
        return locals()[name]

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
