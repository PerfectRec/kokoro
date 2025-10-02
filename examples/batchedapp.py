import modal
from typing import Dict, Any, List, Optional
from pydantic import BaseModel
from pathlib import Path

# Get the parent directory (kokoro root)
kokoro_root = Path(__file__).parent.parent

# Define the image with local kokoro code
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "huggingface_hub",
        "loguru", 
        "misaki[en]>=0.9.4",
        "numpy",
        "torch",
        "transformers",
        "scipy",
        "fastapi[standard]"
    )
    .apt_install(["espeak-ng", "ffmpeg"])
    # Copy local kokoro package into the container
    .copy_local_dir(
        local_path=str(kokoro_root / "kokoro"),
        remote_path="/root/kokoro"
    )
)

# Create the Modal app
app = modal.App("kokoro-tts-batched")

# Define fallback GPUs in order of preference
gpu_list = ["T4", "L4", "A10", "L40S"]


class TTSRequest(BaseModel):
    text: str
    voice: str = "af_heart"
    language: str = "english"


class BatchTTSRequest(BaseModel):
    texts: List[str]
    voice: str = "af_heart"
    language: str = "english"
    speeds: Optional[List[float]] = None  # Optional per-text speeds


class WordTimestamp(BaseModel):
    text: str
    whitespace: str
    start_ts: Optional[float]
    end_ts: Optional[float]


class TTSResponse(BaseModel):
    audio_base64: str
    timestamps: List[WordTimestamp]


class BatchTTSResponse(BaseModel):
    audios_base64: List[str]
    timestamps: List[List[WordTimestamp]]
    text_indices: List[int]  # Which original text each result belongs to


# Global variable to store the model instance
model = None


# Load model function
def load_model():
    import sys
    sys.path.insert(0, "/root")
    
    from kokoro import KPipeline, KModel

    model = KModel(repo_id="hexgrad/Kokoro-82M").to("cuda:0").eval()

    pipeline = KPipeline("a", model=model)
    spanish_pipeline = KPipeline("e", model=model)
    portuguese_pipeline = KPipeline("p", model=model)

    # Load the voices we use
    pipeline.load_single_voice("af_heart")
    pipeline.load_single_voice("am_echo")
    pipeline.load_single_voice("am_fenrir")
    pipeline.load_single_voice("af_bella")
    pipeline.load_single_voice("af_sarah")
    pipeline.load_single_voice("am_puck")
    pipeline.load_single_voice("bf_emma")
    pipeline.load_single_voice("bm_george")

    # Spanish
    spanish_pipeline.load_single_voice("ef_dora")
    spanish_pipeline.load_single_voice("em_alex")
    spanish_pipeline.load_single_voice("em_santa")

    # Brazilian Portuguese
    portuguese_pipeline.load_single_voice("pf_dora")
    portuguese_pipeline.load_single_voice("pm_alex")
    portuguese_pipeline.load_single_voice("pm_santa")

    # G2P fixes
    pipeline.g2p.lexicon.golds["cpu"] = pipeline.g2p.lexicon.golds["CPU"] = "sˌipˌijˈu"
    pipeline.g2p.lexicon.golds["gpu"] = pipeline.g2p.lexicon.golds["GPU"] = "ʤˌipˌijˈu"
    pipeline.g2p.lexicon.golds["arXiv"] = pipeline.g2p.lexicon.golds["archive"]
    pipeline.g2p.lexicon.golds["resourcification"] = "ɹisɔɹsɪfɪkeɪʃən"
    pipeline.g2p.lexicon.golds["los"] = "loʊs"
    pipeline.g2p.lexicon.golds["angeles"] = "ˈænʤəlɪs"
    pipeline.g2p.lexicon.golds["cursed"] = "kɜrst"
    pipeline.g2p.lexicon.golds["Candide"] = pipeline.g2p.lexicon.golds["candide"] = (
        "kæn.did˭"
    )
    del pipeline.g2p.lexicon.silvers["cpus"]

    return pipeline, spanish_pipeline, portuguese_pipeline


# Web endpoint for single TTS generation (backwards compatible)
@app.function(
    image=image,
    gpu=gpu_list,
    scaledown_window=10,
    timeout=1200,
    min_containers=0,
    max_containers=20,
)
@modal.fastapi_endpoint(method="POST", route="/generate")
@modal.concurrent(max_inputs=2)
async def generate_speech(request: TTSRequest):
    import subprocess
    import uuid
    import os
    import base64

    # Load the model once per container
    global model
    if model is None:
        model = load_model()

    text = request.text
    voice = request.voice
    language = request.language

    if not text:
        return {"error": "Please provide text to generate speech"}

    # Use the global model instance
    pipeline, spanish_pipeline, portuguese_pipeline = model

    # Generate speech
    if language == "english":
        generator = pipeline(text, voice=voice)
    elif language == "spanish":
        generator = spanish_pipeline(text, voice=voice)
    elif language == "portuguese":
        generator = portuguese_pipeline(text, voice=voice)
    else:
        generator = pipeline(text, voice=voice)

    mp3_file = f"/tmp/kokoro_tts_out_{uuid.uuid4()}.mp3"

    # Collect word-level timestamps
    all_timestamps = []

    # Use ffmpeg as a pipe
    ffmpeg_cmd = [
        "ffmpeg",
        "-y",
        "-f", "s16le",
        "-ar", "24000",
        "-ac", "1",
        "-i", "pipe:0",
        "-af", "volume=2,alimiter=level_in=1:level_out=0.95:limit=0.95",
        "-codec:a", "libmp3lame",
        "-b:a", "48k",
        "-write_xing", "0",
        "-write_id3v2", "1",
        mp3_file,
    ]

    try:
        with subprocess.Popen(
            ffmpeg_cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        ) as ffmpeg_proc:
            for result in generator:
                audio_bytes = (
                    (result.audio.cpu().numpy() * 32767)
                    .clip(-32768, 32767)
                    .astype("int16")
                    .tobytes()
                )
                ffmpeg_proc.stdin.write(audio_bytes)

                if hasattr(result, "tokens") and result.tokens is not None:
                    for token in result.tokens:
                        all_timestamps.append(
                            WordTimestamp(
                                text=token.text,
                                whitespace=token.whitespace,
                                start_ts=token.start_ts,
                                end_ts=token.end_ts,
                            )
                        )

            ffmpeg_proc.stdin.close()
            ffmpeg_proc.wait()

    except subprocess.CalledProcessError:
        return {"error": "Failed to convert audio to MP3"}

    with open(mp3_file, "rb") as f:
        audio_data = f.read()

    os.remove(mp3_file)

    audio_base64 = base64.b64encode(audio_data).decode("utf-8")

    return TTSResponse(audio_base64=audio_base64, timestamps=all_timestamps)


# NEW: Batched endpoint for processing multiple texts efficiently
@app.function(
    image=image,
    gpu=gpu_list,
    scaledown_window=10,
    timeout=1200,
    min_containers=0,
    max_containers=20,
)
@modal.fastapi_endpoint(method="POST", route="/generate_batch")
@modal.concurrent(max_inputs=1)  # Process one batch at a time for optimal GPU usage
async def generate_speech_batch(request: BatchTTSRequest):
    import subprocess
    import uuid
    import os
    import base64
    import numpy as np

    # Load the model once per container
    global model
    if model is None:
        model = load_model()

    texts = request.texts
    voice = request.voice
    language = request.language
    speeds = request.speeds

    if not texts:
        return {"error": "Please provide texts to generate speech"}

    # Use the global model instance
    pipeline, spanish_pipeline, portuguese_pipeline = model

    # Select pipeline based on language
    if language == "english":
        selected_pipeline = pipeline
    elif language == "spanish":
        selected_pipeline = spanish_pipeline
    elif language == "portuguese":
        selected_pipeline = portuguese_pipeline
    else:
        selected_pipeline = pipeline

    # Use batched generation - this is the key improvement!
    results = selected_pipeline.generate_batch(
        texts=texts,
        voice=voice,
        speed=speeds if speeds else 1.0
    )

    # Process each result to MP3
    audios_base64 = []
    all_timestamps = []
    text_indices = []

    for result in results:
        mp3_file = f"/tmp/kokoro_tts_batch_{uuid.uuid4()}.mp3"
        
        # Convert to MP3 using ffmpeg
        ffmpeg_cmd = [
            "ffmpeg",
            "-y",
            "-f", "s16le",
            "-ar", "24000",
            "-ac", "1",
            "-i", "pipe:0",
            "-af", "volume=2,alimiter=level_in=1:level_out=0.95:limit=0.95",
            "-codec:a", "libmp3lame",
            "-b:a", "48k",
            "-write_xing", "0",
            "-write_id3v2", "1",
            mp3_file,
        ]

        try:
            with subprocess.Popen(
                ffmpeg_cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            ) as ffmpeg_proc:
                audio_bytes = (
                    (result.audio.cpu().numpy() * 32767)
                    .clip(-32768, 32767)
                    .astype("int16")
                    .tobytes()
                )
                ffmpeg_proc.stdin.write(audio_bytes)
                ffmpeg_proc.stdin.close()
                ffmpeg_proc.wait()

            # Read and encode MP3
            with open(mp3_file, "rb") as f:
                audio_data = f.read()
            os.remove(mp3_file)
            
            audios_base64.append(base64.b64encode(audio_data).decode("utf-8"))

            # Collect timestamps
            timestamps = []
            if hasattr(result, "tokens") and result.tokens is not None:
                for token in result.tokens:
                    timestamps.append(
                        WordTimestamp(
                            text=token.text,
                            whitespace=token.whitespace,
                            start_ts=token.start_ts,
                            end_ts=token.end_ts,
                        )
                    )
            all_timestamps.append(timestamps)
            
            # Track which original text this result came from
            text_indices.append(result.text_index if result.text_index is not None else 0)

        except subprocess.CalledProcessError:
            return {"error": f"Failed to convert audio to MP3 for result {len(audios_base64)}"}

    return BatchTTSResponse(
        audios_base64=audios_base64,
        timestamps=all_timestamps,
        text_indices=text_indices
    )


# Health check endpoint
@app.function(image=image)
@modal.fastapi_endpoint(method="GET", route="/health")
async def health_check():
    return {"status": "healthy", "batching": "enabled"}


# To run the ephemeral endpoint: modal serve examples/modal_batched.py
# To deploy: modal deploy examples/modal_batched.py

