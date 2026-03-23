from bricks2marble.tf import AnnotationHMM
from hidten import HMMMode


class HMMBlock(AnnotationHMM):
    def __init__(
        self,
        mode: HMMMode,
        parallel: int,
        training: bool,
        emitter_epsilon: float = 0.01,
        initial_exon_len: int = 200,
        initial_intron_len: int = 4500,
        initial_ir_len: int = 10000,
    ) -> None:
        self.mode = mode
        self.parallel = parallel
        self.training = training
        super().__init__(
            use_reverse_strand=False,
            emitter_eye=emitter_epsilon,
            train_emitter=False,
            initial_exon_len=initial_exon_len,
            initial_intron_len=initial_intron_len,
            initial_ir_len=initial_ir_len,
            transitioner_share_frames=False,
            transitioner_share_noncoding=False,
            train_transitions=False,
            train_start_dist=False
        )

    def call(self, x, nuc):
        return super().call(
            x,
            nuc,
            mode=self.mode,
            parallel=self.parallel,
            training=self.training,
        )
