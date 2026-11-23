# GroundingDINO CUDA extension for RTX 50-series (sm_120)

The GroundingDINO extension (`groundingdino/_C`) was built without `TORCH_CUDA_ARCH_LIST`, so it
only contained kernels for the build machine's GPU (sm_89). On Blackwell (sm_120) every
deformable-attention call fails with `no kernel image is available for execution on the device`.

Getting it to work on sm_120 needs:
1. **PyTorch built with CUDA ≥ 12.8** (e.g. torch 2.7.1+cu128). torch 2.1.2+cu121 (`dino_real`) can't run on Blackwell at all.
2. **A source fix** (`sm120_csrc.patch`). The old code used `tensor.type()`, which newer PyTorch rejects:
   - `AT_DISPATCH_FLOATING_TYPES(value.type(), ...)` → `value.scalar_type()`
   - `x.type().is_cuda()` → `x.is_cuda()`
   - `.data<T>()` → `.data_ptr<T>()`

   These changes also compile on torch 2.1.
3. **Building with** `TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9;12.0+PTX"`.

## Contents
| Path | What |
|---|---|
| `src/` | GroundingDINO package source with the patch applied (no weights/data) |
| `sm120_csrc.patch` | diff of `csrc/` vs. the original repo (`diff -r original patched`) |
| `setup_env.sh` | creates the conda env, builds, verifies arches, runs the kernel test |
| `tests/test_msda.py` | compiled kernel vs. pure-PyTorch reference (fwd + bwd, fp32/fp64) + timing |
| `tests/e2e_detect.py` | full model on one frame; dumps boxes to JSON (weights path is hardcoded to `../GroundingDINO/weights`) |
| `results/` | e2e outputs from `dino_real` (old build) vs. `dino_sm120` (new build) on `frame.jpg` |
| `build_sm120.log` | full build log |

## Usage
```bash
bash setup_env.sh            # creates env "dino_sm120"; pass another name as arg 1
conda activate dino_sm120
python tests/test_msda.py    # should print ALL PASS
python tests/e2e_detect.py results/frame.jpg out.json
```

## Results (2026-11-23, RTX 6000 Ada, sm_89)
- Built `_C.so`: SASS sm_80/86/89/120, PTX compute_120.
- Kernel matches the PyTorch reference (fp32 fwd err 4e-9, grads rel err ≤1e-6). CUDA kernel takes 0.018 ms vs 0.275 ms for the PyTorch fallback.
- End to end (SwinT-OGC): same detection as the old build. Confidence 0.6583 vs 0.6585, box coordinates differ by at most 7e-7, ~75 ms/frame in both.
- **Not yet run on an actual sm_120 GPU.** That needs to be validated on the RTX 5090.
