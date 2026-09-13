<div align="center">

# FixAnything: 3D-Consistent Rendering Refinement via Video Generative Priors

[Khiem Vuong](https://www.khiemvuong.com/), [Deva Ramanan*](https://www.cs.cmu.edu/~deva), [Srinivasa Narasimhan*](https://www.cs.cmu.edu/~srinivas)

**ECCV 2026**

[[`arXiv`](https://arxiv.org/abs/2608.23549)]
[[`Project Page`](https://fix-anything.github.io/)]
[[`Model (HF)`](https://huggingface.co/kvuong2711/fix-anything)]
[[`Bibtex`](#-citation)]

</div>

<p align="center">
  <img src="assets/teaser.png" width="100%">
</p>

## 📖 Overview

FixAnything is a single generalist video model that repairs rendering artifacts from **any** 3D representation — 3DGS, NeRF, meshes, or sparse point clouds — by repurposing a pretrained video diffusion model with minimal modification and finetuning. 

This repository contains the inference code.

## Table of Contents

- [Installation](#-installation)
- [Checkpoints](#-checkpoints)
- [Run Inference on a Rendered Video](#-run-inference-on-a-rendered-video)
- [Run on Your Own Captures](#-run-on-your-own-captures)
- [License](#-license)
- [Acknowledgments](#-acknowledgments)
- [Issues](#-issues)
- [Citation](#-citation)

## 🔧 Installation

```bash
git clone https://github.com/kvuong2711/fix-anything.git
cd fix-anything

conda create -n fixanything python=3.10 -y
conda activate fixanything

# PyTorch for your CUDA version
pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu126

# FixAnything + the pinned DiffSynth-Studio commit it builds on
# (DiffSynth's setup.py needs pkg_resources, hence setuptools<82 and --no-build-isolation)
pip install "setuptools<82" wheel
pip install --no-build-isolation -e .

# Optional: MapAnything + pyrender, to reconstruct and render your own captures (see below)
pip install --no-build-isolation -e ".[mapanything]"

# Optional but recommended: FlashAttention-2 (used automatically when installed; prebuilt wheel for torch 2.6 / CUDA 12 / Python 3.10)
pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.6cxx11abiTRUE-cp310-cp310-linux_x86_64.whl
```

## 📦 Checkpoints

`scripts/download_models.py` downloads the lightest working setup (~22 GB) into `models/`: the Wan2.1-I2V-14B-480P VAE and CLIP image encoder, the FixAnything LoRA from [kvuong2711/fix-anything](https://huggingface.co/kvuong2711/fix-anything), and the fp8_e4m3fn single-file repack of the DiT from [Comfy-Org/Wan_2.1_ComfyUI_repackaged](https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged) (16 GB):

```bash
python scripts/download_models.py                  # fp8 DiT (default, 16 GB)
python scripts/download_models.py --dit bf16       # Comfy-Org bf16 repack (33 GB)
python scripts/download_models.py --dit original   # Wan-AI's sharded bf16 release (33 GB, also via --source modelscope)
```

The fp8 weights are only a storage format: compute stays in bf16, the LoRA is applied unmerged (so it is not rounded to fp8) and the output is visually equivalent to bf16. They halve the download, the disk footprint and the CPU RAM needed while the model is offloaded, which matters on pods with a ~50 GB RAM limit. `run_inference.py` picks up whichever DiT is present (`--dit <file>` selects one explicitly).

The umT5-XXL text encoder (11 GB) is not downloaded: FixAnything uses fixed prompts, and their embeddings are shipped in the repo as `models/prompt_embeds.pt` (git LFS, ~13 MB), which `run_inference.py` uses automatically. The resulting layout is:

```
models/
├── prompt_embeds.pt                                   (from the repo, via git LFS)
├── fixanything_lora.safetensors
└── Wan-AI/Wan2.1-I2V-14B-480P/
    ├── Wan2.1_VAE.pth
    ├── models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth
    └── split_files/diffusion_models/wan2.1_i2v_480p_14B_fp8_e4m3fn.safetensors   (or the bf16 repack / original shards)
```

If you already have the Wan2.1 files elsewhere, point `--model_dir` to the folder that contains `Wan-AI/`.

On a fresh RunPod pod, `scripts/setup_runpod.sh` does all of the above (clone, venv, packages, download) in one go; `DIT=bf16 bash scripts/setup_runpod.sh` selects the bf16 DiT.

### Optional: custom prompts

To use other prompts, download the text encoder (`python scripts/download_models.py --text_encoder`) and either encode them once (`python scripts/encode_prompts.py --prompt "..." --negative_prompt "..."`; `prompt_embeds.pt` is merged, not overwritten) or pass `--prompt_embeds none` to `run_inference.py` to run the text encoder at inference time.

## 🚀 Run Inference on a Rendered Video

`run_inference.py` refines a rendered video (a video file, or a folder of frames). `examples/` contains one DL3DV clip rendered by 3D Gaussian Splatting and by a sparse point-track renderer:

```bash
# 3DGS rendering
python scripts/run_inference.py --input examples/dl3dv_3dgs/input.mp4   --output_dir outputs/dl3dv_3dgs

# sparse point-track rendering
python scripts/run_inference.py --input examples/dl3dv_tracks/input.mp4 --output_dir outputs/dl3dv_tracks
```

Each run writes `generated.mp4` (refined video), `input.mp4` (the resized input) and `side_by_side.mp4` (input | refined) to the output folder.

To run on your own rendering (any 3D representation), pass a video of 61 frames rendered along a camera path, ideally starting and ending at views used to build the representation. Frames are resized to 832×480.

- **Caveat on the number of frames.** Internally the last frame is repeated 4 times (65 frames): the Wan VAE encodes the first frame on its own but compresses the rest temporally, so this (hopefully) gives the last frame more influence to the generation. The output is cropped back to 61 frames.
- **`--clean_frame_indices`** lists the frames to keep as-is; by default the first and last frames (`"0 60"`).

## 📷 Run on Your Own Captures

If you only have a few photos of a scene, `scripts/run_mapanything.py` reconstructs them with [MapAnything](https://github.com/facebookresearch/map-anything), renders the reconstruction along a camera path from the first to the last image, and FixAnything turns that rendering into a clean video. `examples/microkitchen_2views` and `examples/chair_2views` each contain two photos of a scene (credit: thanks to Peter Hedman for the photos!):

```bash
# 1) MapAnything reconstruction + rendering (61 frames; first/last frame = the input photos)
python scripts/run_mapanything.py --images examples/microkitchen_2views/images --output_dir outputs/microkitchen_2views

# 2) FixAnything
python scripts/run_inference.py --input outputs/microkitchen_2views/rendered.mp4 --output_dir outputs/microkitchen_2views
```

Step 1 writes `rendered.mp4` and `reconstruction.glb`; step 2 adds `generated.mp4` (the result) and `side_by_side.mp4`.

- `--trajectory`: `interpolate` (default) moves the camera on a straight path between the two input views; `push_in` additionally dollies toward the scene mid-path and back.
- `--render`: render the reconstruction as a triangle `mesh` (default) or as a point cloud (`pcd`).

Example on `examples/microkitchen_2views` with `--trajectory push_in`:

<p align="center">
  <img src="assets/microkitchen_push_in.gif" width="100%">
  <br>
  <em>MapAnything rendering (left), FixAnything output (right).</em>
</p>

## 📄 License

The code and the FixAnything weights are released under the [Apache 2.0 License](LICENSE), the license of the underlying [Wan2.1](https://github.com/Wan-Video/Wan2.1) model.

## 🙏 Acknowledgments

This codebase builds upon many excellent open-source projects, such as [DiffSynth-Studio](https://github.com/modelscope/DiffSynth-Studio), [Wan2.1](https://github.com/Wan-Video/Wan2.1), [MapAnything](https://github.com/facebookresearch/map-anything), [gsplat](https://github.com/nerfstudio-project/gsplat), [DL3DV](https://dl3dv-10k.github.io/DL3DV-10K/), etc. We thank the respective authors for making their work publicly available.

## 🐛 Issues

If you have any problem/question/suggestion, feel free to create an issue or reach out directly to me via [email](mailto:kvuong@andrew.cmu.edu).

## 📝 Citation

```bibtex
@inproceedings{vuong2026fixanything,
  title     = {FixAnything: 3D-Consistent Rendering Refinement via Video Generative Priors},
  author    = {Vuong, Khiem and Ramanan, Deva and Narasimhan, Srinivasa},
  booktitle = {European Conference on Computer Vision},
  year      = {2026}
}
```
