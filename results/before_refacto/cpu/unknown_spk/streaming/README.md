Repo: https://github.com/juanmc2005/diart

Main code:
```
from diart.inference import Benchmark
from diart import OnlineSpeakerDiarization, PipelineConfig
from diart.models import SegmentationModel


config = PipelineConfig(
    # Set the model used in the paper
    # segmentation=SegmentationModel.from_pyannote("pyannote/segmentation"),
    step=0.5,
    latency=1.5,
    tau_active=0.555,
    rho_update=0.422,
    delta_new=1.517
)
pipeline = OnlineSpeakerDiarization(config)
benchmark = Benchmark("audio", 
    "rttm",
    "output")
benchmark(pipeline)
```

Could be tested with:
# segmentation = SegmentationModel.from_pyannote("pyannote/segmentation@Interspeech2021"),
# segmentation = SegmentationModel.from_pyannote("/home/wghezaiel/Project_2/pyBK_version2/diarizer/models/pyannote/alimeeting_epoch0_step2492.ckpt"),
