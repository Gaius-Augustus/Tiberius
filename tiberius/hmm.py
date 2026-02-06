from bricks2marble.tf import AnnotationHMM
from hidten import HMMMode


class HMMBlock(AnnotationHMM):

    def __init__(self, mode: HMMMode, parallel: int, training: bool) -> None:
        self.mode = mode
        self.parallel = parallel
        self.training = training
        super().__init__(
            use_reverse_strand=False,
            train_emitter=False,
            train_transitioner=False,
            initial_exon_len=1000,
            initial_intron_len=10_000,
            initial_ir_len=10_000,
        )

    def call(self, x, nuc):
        super().call(
            x,
            nuc,
            mode=self.mode,
            parallel=self.parallel,
            training=self.training,
        )
