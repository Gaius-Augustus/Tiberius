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
        train_emitter: bool= False,
        transitioner_share_frames: bool = False,
        transitioner_share_noncoding: bool = False,
        train_transitions: bool = False,
        train_start_dist: bool = False

    ) -> None:
        self.mode = mode
        self.parallel = parallel
        self.training = training
        super().__init__(
            use_reverse_strand=False,
            emitter_eye=emitter_epsilon,
            train_emitter=train_emitter,
            initial_exon_len=initial_exon_len,
            initial_intron_len=initial_intron_len,
            initial_ir_len=initial_ir_len,
            transitioner_share_frames=transitioner_share_frames,
            transitioner_share_noncoding=transitioner_share_noncoding,
            train_transitions=train_transitions,
            train_start_dist=train_start_dist
        )

    def call(self, x, nuc):
        return super().call(
            x,
            nuc,
            mode=self.mode,
            parallel=self.parallel,
            training=self.training,
        )
