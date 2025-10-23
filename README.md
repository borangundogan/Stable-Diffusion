# 🧠 Stable Diffusion from Scratch — Project Configuration (YAML Overview)

Below is the complete **project specification in YAML format**, including environment setup, dependencies, structure, run commands, and notes.  

---

```yaml
project:
  name: stable_diffusion_from_scratch
  description: >
    Build Stable Diffusion completely from scratch — including VAE, CLIP text encoder,
    U-Net, DDPM scheduler, and inference pipeline. Designed for learning and understanding
    each internal component.

  python: "3.11.3"

  dependencies:
    - torch
    - numpy
    - tqdm
    - transformers
    - lightning
    - pillow
    - torchvision
    - torchmetrics
    - accelerate
    - safetensors
    - diffusers
    - matplotlib

  structure:
    - vae/: "Variational Autoencoder implementation"
    - clip/: "CLIP text encoder"
    - unet/: "U-Net noise prediction model"
    - scheduler/: "DDPM scheduler (noise/denoising steps)"
    - pipeline/: "End-to-end diffusion pipeline"
    - inference/: "Run inference to generate images"
    - utils/: "Configs, visualization tools"
    - main.py: "Entry point script"
    - pyproject.toml: "uv environment configuration"

  setup:
    instructions: |
      1. Initialize project with uv:
         uv init stable_diffusion_from_scratch
         cd stable_diffusion_from_scratch
      2. Pin Python version:
         uv python pin 3.11.3
      3. Install dependencies:
         uv add torch numpy tqdm transformers lightning pillow torchvision torchmetrics accelerate safetensors diffusers matplotlib

  run:
    examples:
      - "uv run python main.py"
      - "uv run python vae/vae_model.py"
      - "uv run python pipeline/diffusion_pipeline.py"

  notes: |
    - Start development in this order: VAE → CLIP → U-Net → Scheduler → Pipeline.
    - Test each module independently before integrating.
    - Use matplotlib or utils/visualize.py to visualize diffusion steps.
    - Check GPU availability using:
        uv run python -c "import torch; print(torch.cuda.is_available())"

  references:
    - "https://arxiv.org/abs/2112.10752  # Stable Diffusion"
    - "https://arxiv.org/abs/2006.11239  # DDPM"
    - "https://arxiv.org/abs/2103.00020  # CLIP"

  goal: "Understand and re-create how text-to-image diffusion models work internally."
