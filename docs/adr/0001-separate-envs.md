# ADR 0001: Separate environments for training and serving

Status: accepted, 26 Oct 2026

## Context

Training uses Unsloth (LoRA on a vision-language model). Serving uses vLLM. Both pin
narrow ranges of torch and transformers: vLLM 0.32.0 requires
`transformers>=5.16.1,<5.20.0`, and Unsloth pins the versions it patches. The RTX 5090
(Blackwell, compute capability 12.0) also needs CUDA 12.8 or newer builds of every
compiled package.

## Decision

Keep two uv environments, `.venv-train` and `.venv-serve`, created by
`scripts/stage1/setup_envs.sh`. The only thing shared between them is the LoRA adapter on
disk, in the standard PEFT format. Each environment's exact versions are frozen to
`requirements/*-freeze.txt` once stage 1 passes on the GPU.

## Consequences

- Neither tool runs on versions its authors never tested, and upgrading one cannot break
  the other.
- The adapter format is the contract. The serving smoke test checks it end to end: train,
  save, load into vLLM, serve under load.
- The cost is disk space (two copies of torch and the CUDA libraries, about 10 GB each)
  and two installs.
- Alternative rejected: one environment. It saves disk but makes every upgrade a
  dependency negotiation between two fast-moving projects.
