"""Audio transport: representation, measurement, rate conversion, buffering, VAD, devices."""

from .buffer import PreRoll, RingBuffer
from .io import (
    AudioInput,
    AudioOutput,
    CapturingOutput,
    DeviceInfo,
    SounddeviceInput,
    SounddeviceOutput,
    SyntheticInput,
    WavFileInput,
    list_devices,
)
from .pcm import (
    crossfade,
    fade_edges,
    from_float_samples,
    gain,
    measure,
    normalize_peak,
    peak,
    remove_dc,
    rms,
    to_float_samples,
)
from .preprocess import PreprocessReport, Preprocessor, trim_silence
from .resample import MODE_BAND_LIMITED, MODE_LINEAR, resample
from .vad import (
    EnergySpeechDetector,
    NeuralSpeechDetectorUnavailable,
    SegmenterConfig,
    SpeechActivityDetector,
    SpeechSegment,
    UtteranceSegmenter,
    installed_detector,
)
from .wav import WavInfo, inspect as inspect_wav, read_wav, write_wav

__all__ = [
    "AudioInput",
    "AudioOutput",
    "CapturingOutput",
    "DeviceInfo",
    "EnergySpeechDetector",
    "MODE_BAND_LIMITED",
    "MODE_LINEAR",
    "NeuralSpeechDetectorUnavailable",
    "PreRoll",
    "PreprocessReport",
    "Preprocessor",
    "RingBuffer",
    "SegmenterConfig",
    "SounddeviceInput",
    "SounddeviceOutput",
    "SpeechActivityDetector",
    "SpeechSegment",
    "SyntheticInput",
    "UtteranceSegmenter",
    "WavFileInput",
    "WavInfo",
    "crossfade",
    "fade_edges",
    "from_float_samples",
    "gain",
    "inspect_wav",
    "installed_detector",
    "list_devices",
    "measure",
    "normalize_peak",
    "peak",
    "read_wav",
    "remove_dc",
    "resample",
    "rms",
    "to_float_samples",
    "trim_silence",
    "write_wav",
]
