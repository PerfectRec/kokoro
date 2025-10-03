import os
import wave
import numpy as np

from kokoro import KPipeline


def float_to_int16(audio: np.ndarray) -> bytes:
    audio = np.clip(audio, -1.0, 1.0)
    return (audio * 32767.0).astype('<i2').tobytes()


def save_wav(path: str, audio: np.ndarray, sample_rate: int = 24000):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with wave.open(path, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(float_to_int16(audio))


def main():
    texts = [
        "Hello, this is a short batched synthesis test.",
        "Batched inference with fp16 and frame masking should be efficient.",
        "Kokoro TTS generates natural speech from text.",
        "The quick brown fox jumps over the lazy dog.",
    ]

    pipeline = KPipeline(lang_code='a')
    results = pipeline.generate_batch(texts, voice='af_heart', speed=1.0)

    out_dir = os.path.join(os.path.dirname(__file__), 'outputs')
    for i, r in enumerate(results):
        wav_path = os.path.join(out_dir, f'batch_{i}.wav')
        save_wav(wav_path, r.audio.numpy())
        print(f"Saved: {wav_path}")


if __name__ == '__main__':
    main()


