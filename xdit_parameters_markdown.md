# xDiT Parameters Explained

## 1. Command Line Arguments (`sys.argv`)

These are passed to `xFuserArgs` when configuring the xDiT engine:

```python
sys.argv = [
    'jupyter',
    '--model', model_id,
    '--height', str(height),
    '--width', str(width),
    '--num_inference_steps', str(steps),
    '--prompt', prompt,
    '--seed', str(seed),
    '--data_parallel_degree', '1',
    '--ulysses_degree', '1',
    '--ring_degree', '1',
    '--pipefusion_parallel_degree', '1',
    '--tensor_parallel_degree', '1',
]
```

---

## 2. Basic Parameters

| Parameter | Description |
|----------|-------------|
| `--model` | HuggingFace model ID to load (e.g., `"stabilityai/stable-diffusion-3-medium"`) |
| `--height` | Output image height in pixels (e.g., `1024`) |
| `--width` | Output image width in pixels (e.g., `1024`) |
| `--num_inference_steps` | Number of denoising steps — higher = better quality but slower |
| `--prompt` | Text description of the image to generate |
| `--seed` | Random seed for reproducible results |

---

## 3. Parallelization Parameters (Advanced)

These control how xDiT distributes computation across GPUs. **For a single GPU, all should remain `1`.**

| Parameter | Description | Single GPU Value |
|-----------|-------------|------------------|
| `--data_parallel_degree` | Splits batches across GPUs (data parallelism) | `1` |
| `--ulysses_degree` | Splits sequence dimension in attention layers | `1` |
| `--ring_degree` | Ring attention for long sequences | `1` |
| `--pipefusion_parallel_degree` | Pipeline parallelism across diffusion steps | `1` |
| `--tensor_parallel_degree` | Splits individual tensors across GPUs | `1` |

### Multi‑GPU Example
- 2 GPUs: `--data_parallel_degree 2`
- 4 GPUs: `--pipefusion_parallel_degree 2 --tensor_parallel_degree 2`

---

## 4. Configuration Objects

### `engine_config`
Created from `xFuserArgs`:

```python
engine_args = xFuserArgs.from_cli_args(args)
engine_config, input_config = engine_args.create_config()
```

Stores:
- GPU topology
- Parallelization degrees
- Distributed settings

### `input_config`
Contains the generation parameters:
- `height`
- `width`
- `prompt`
- `num_inference_steps`
- `seed`
- `output_type` (`pil`, `latent`, `np`)

---

## 5. xDiTParallel Wrapper

```python
from xfuser import xDiTParallel
pipe = xDiTParallel(pipe, engine_config, input_config)
```

### Purpose
Wraps your diffusion pipeline to enable xDiT’s optimizations:
- Distributed inference across GPUs
- Applies parallelization strategies
- Manages GPU-to-GPU communication

### Parameters
- `pipe` — diffusion pipeline (e.g., `StableDiffusion3Pipeline`)
- `engine_config` — parallelization strategy
- `input_config` — generation parameters

---

## 6. Environment Variables (set **before** importing `xfuser`)

```python
os.environ['RANK'] = '0'
os.environ['WORLD_SIZE'] = '1'
os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = '29500'
os.environ['LOCAL_RANK'] = '0'
```

These configure PyTorch distributed mode, even on a single GPU.

