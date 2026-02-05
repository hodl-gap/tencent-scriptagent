# The Script is All You Need: An Agentic Framework for Long-Horizon Dialogue-to-Cinematic Video Generation

## Overview
<p align="center">
  <img src="./figures/overview.png" alt="Framework Overview" width="100%">
</p>

Our framework consists of three key agents working in concert:

1. **ScriptAgent**: Transforms coarse-grained dialogues into detailed shooting scripts
2. **DirectorAgent**: Generates multi-shot videos from scripts while maintaining visual continuity
3. **CriticAgent**: Evaluates both script quality and video generation fidelity

## Setup

### 1. Clone and configure API keys

```bash
git clone <repo-url> && cd tencent_scriptagent
cp .env.example .env
# Edit .env and fill in your API keys
```

The `.env` file is loaded automatically by `pipeline.py`. You need at least one pair of keys depending on which video model you use:

| Key | What uses it | Where to get it |
|-----|-------------|-----------------|
| `OPENAI_API_KEY` | Sora video generation, script evaluation (GPT-4o) | [platform.openai.com](https://platform.openai.com/api-keys) |
| `GEMINI_API_KEY` | Veo video generation, video evaluation (Gemini 2.5 Pro) | [aistudio.google.com](https://aistudio.google.com/apikey) |

### 2. Install Python dependencies

Core dependencies (always needed):

```bash
pip install python-dotenv opencv-python moviepy pillow
```

Then install the SDK for your chosen video model:

```bash
# For Veo (default)
pip install google-genai

# For Sora
pip install openai

# For both (needed if running evaluation too)
pip install openai google-genai
```

### 3. Get a shooting script

The pipeline requires a **shooting script** as input to DirectorAgent. You have three options:

**Option A: Use the online demo (no GPU needed)**
Generate a script at [huggingface.co/spaces/XD-MU/ScriptAgent](https://huggingface.co/spaces/XD-MU/ScriptAgent), copy the output into a `.txt` file, then pass it via `--script_path`.

**Option B: Let ScriptAgent generate it locally (requires GPU)**
ScriptAgent runs a local LLM to convert dialogue into a shooting script. This requires:
- A CUDA GPU with sufficient VRAM
- The model weights downloaded from HuggingFace:
  ```bash
  pip install ms-swift[llm] huggingface-hub
  python -c "
  from huggingface_hub import snapshot_download
  snapshot_download('XD-MU/ScriptAgent', local_dir='./models/ScriptAgent',
                    local_dir_use_symlinks=False, resume_download=True)
  "
  ```
  When you omit `--script_path`, the pipeline runs ScriptAgent automatically.

**Option C: Write one by hand**
See [Script Format](#script-format) below.

## Quick Start

### Full pipeline (with pre-generated script)

```bash
python code/pipeline.py \
    --dialogue "Your dialogue text or path to .txt file" \
    --script_path ./my_script.txt \
    --output_dir ./output
```

API keys are read from `.env` automatically. This runs: DirectorAgent (Veo) then CriticAgent (script + video evaluation).

### Full pipeline (ScriptAgent generates the script)

Requires GPU and downloaded model weights (see Step 3 above):

```bash
python code/pipeline.py \
    --dialogue "Your dialogue text or path to .txt file" \
    --output_dir ./output
```

### Skip evaluation

```bash
python code/pipeline.py \
    --dialogue "input dialogue" \
    --script_path ./my_script.txt \
    --skip_eval
```

Only `GEMINI_API_KEY` is needed here (for Veo). No `OPENAI_API_KEY` required.

### Use Sora instead of Veo

```bash
python code/pipeline.py \
    --dialogue "input dialogue" \
    --script_path ./my_script.txt \
    --video_model sora2-pro \
    --skip_eval
```

Only `OPENAI_API_KEY` is needed here. No `GEMINI_API_KEY` required.

### Pipeline arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--dialogue` | Input dialogue text, or path to a `.txt` file **(required)** | — |
| `--script_path` | Path to a pre-generated script file (skips ScriptAgent) | `None` |
| `--video_model` | Video model: `veo3.1`, `veo3.1-fast`, `sora2-pro`, `sora2` | `veo3.1` |
| `--style` | Visual style: `anime`, `realistic`, `animated`, `painterly`, `abstract` | `anime` |
| `--output_dir` | Output directory | `./output` |
| `--model_path` | ScriptAgent model weights path | `./models/ScriptAgent` |
| `--skip_eval` | Skip CriticAgent evaluation | `false` |
| `--openai_api_key` | Override `OPENAI_API_KEY` from `.env` | from env |
| `--gemini_api_key` | Override `GEMINI_API_KEY` from `.env` | from env |

### Video generation models

Each model produces independent output — you pick one per run. Only the API key for that model's service is needed for video generation.

| Model | Service | Key |
|-------|---------|-----|
| `veo3.1`, `veo3.1-fast` | Google Veo (Gemini API) | `GEMINI_API_KEY` |
| `sora2-pro`, `sora2` | OpenAI Sora | `OPENAI_API_KEY` |

Wan2.5, Kling, Vidu, and Jimeng are listed in `MODEL_DEFAULT_CONFIG` but not yet wired to their public APIs (each needs its own SDK). PRs welcome.

---

## Script Format

DirectorAgent expects a shooting script with these sections:

```
【Character Description】
Alice: A young woman with long brown hair, wearing a blue dress.
Bob: An elderly man with white beard, in formal suit.

【Scene Description】
A sunny afternoon in a beautiful garden with blooming flowers.

【Character Positions】
1. Alice stands on the left, Bob on the right
2. Both move to the center
3. Alice sits on bench, Bob stands nearby

【Dialogue】
1. Alice: "What a beautiful day!" (smiling and looking around)
2. Bob: "Indeed, reminds me of my youth." (nostalgic expression)
3. Alice: "Tell me more!" (sitting down, eager to listen)
```

---

## Running Individual Agents

### ScriptAgent

- **Model**: [XD-MU/ScriptAgent](https://huggingface.co/XD-MU/ScriptAgent)
- **Online Demo**: [Try ScriptAgent](https://huggingface.co/spaces/XD-MU/ScriptAgent)
- **Project Page**: [The Script is All You Need](https://xd-mu.github.io/ScriptIsAllYouNeed/)

```python
import os
from huggingface_hub import snapshot_download
from swift.llm import PtEngine, RequestConfig, InferRequest
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

model_name = "XD-MU/ScriptAgent"
local_path = "./models/ScriptAgent"

snapshot_download(
    repo_id=model_name,
    local_dir=local_path,
    local_dir_use_symlinks=False,
    resume_download=True
)

engine = PtEngine(local_path, max_batch_size=1)
request_config = RequestConfig(max_tokens=8192, temperature=0.7)

infer_request = InferRequest(messages=[
    {"role": "user", "content": "Your Dialogue"}
])
response = engine.infer([infer_request], request_config)[0]
print(response.choices[0].message.content)
```

### DirectorAgent

```bash
# Veo
python code/director_agent.py \
    --script_path ./storyscript.txt \
    --model veo3.1 \
    --style anime \
    --output_dir ./output

# Sora
python code/director_agent.py \
    --script_path ./storyscript.txt \
    --model sora2-pro \
    --style anime \
    --output_dir ./output
```

API keys are read from `GEMINI_API_KEY` / `OPENAI_API_KEY` env vars (or `.env` file), or passed via `--gemini_api_key` / `--openai_api_key`.

DirectorAgent configuration:

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--model` | Video model | `sora2-pro` |
| `--style` | Visual style | `anime` |
| `--size` | Video resolution | `1792x1024` |
| `--seconds` | Duration per shot | `12` |
| `--reference_mode` | Reference type: `first`, `style`, `asset` | `first` |
| `--max_retry` | Max retry on failure | `5` |

### CriticAgent — Script Evaluation

```bash
python code/critic_agent_script.py \
    --scripts_jsonl infer_script_result.jsonl \
    --dialogues_json test.json \
    --output_json evaluation_results/script_eval.json \
    --api_key $OPENAI_API_KEY \
    --model gpt-4o
```

### CriticAgent — Video Evaluation

**Gemini backend (recommended):**

```bash
python code/critic_agent_video.py \
    --backend gemini \
    --video_folder output_story/sora2-pro/final_video \
    --mapping_jsonl video_dialogues.jsonl \
    --output_json evaluation_results/video_eval_gemini.json \
    --api_key $GEMINI_API_KEY \
    --model gemini-2.5-pro
```

**Qwen3-Omni backend (local, no API cost):**

```bash
python code/critic_agent_video.py \
    --backend qwen \
    --video_folder output_story/sora2-pro/final_video \
    --mapping_jsonl video_dialogues.jsonl \
    --output_json evaluation_results/video_eval_qwen.json \
    --device cuda
```

---

## Citation

If you find this work useful, please cite:

```bibtex
@misc{mu2026scriptneedagenticframework,
      title={The Script is All You Need: An Agentic Framework for Long-Horizon Dialogue-to-Cinematic Video Generation},
      author={Chenyu Mu and Xin He and Qu Yang and Wanshun Chen and Jiadi Yao and Huang Liu and Zihao Yi and Bo Zhao and Xingyu Chen and Ruotian Ma and Fanghua Ye and Erkun Yang and Cheng Deng and Zhaopeng Tu and Xiaolong Li and Linus},
      year={2026},
      eprint={2601.17737},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2601.17737},
}
```

## Acknowledgments
- Thanks to [VBench](https://github.com/Vchitect/VBench) for providing video evaluation metrics.
- Thanks to [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) for providing the SFT training framework.
- Thanks to [ms-swift](https://github.com/modelscope/ms-swift) for providing the GRPO training framework.
