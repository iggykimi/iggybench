# vLLM 0.32.0 plus the audio libraries the transcription models need (soundfile, librosa, mistral-common).
# Built automatically by serve.sh for the configs that set BUILD to this folder. Package versions were not pinned.
FROM vllm/vllm-openai:v0.32.0
RUN (python3 -m pip install --no-cache-dir soundfile "mistral-common[soundfile]" librosa \
     || uv pip install --system soundfile "mistral-common[soundfile]" librosa) \
 && (python3 -c "import soundfile" || (apt-get update && apt-get install -y --no-install-recommends libsndfile1)) \
 && python3 -c "import soundfile, librosa, mistral_common; print('audio OK', soundfile.__libsndfile_version__)"
