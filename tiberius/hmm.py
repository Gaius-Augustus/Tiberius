from bricks2marble.tf import AnnotationHMM
from hidten import HMMMode


class HMMBlock(AnnotationHMM):

    def __init__(
        self,
        mode: HMMMode,
        parallel: int,
        training: bool,
        emitter_epsilon: float = 0.0,
    ) -> None:
        self.mode = mode
        self.parallel = parallel
        self.training = training
        super().__init__(
            use_reverse_strand=False,
            emitter_eye=emitter_epsilon,
            train_emitter=False,
            initial_exon_len=200,
            initial_intron_len=4500,
            initial_ir_len=10_000,
            transitioner_share_frames=False,
            transitioner_share_noncoding=False,
            train_transitioner=False,
        )

    def call(self, x, nuc):
        return super().call(
            x,
            nuc,
            mode=self.mode,
            parallel=self.parallel,
            training=self.training,
        )
