## Work With Me

I offer PyTorch/CUDA GPU performance optimization services, including:

- GPU performance audits
- PyTorch inference optimization
- CUDA bottleneck analysis
- FP16 / mixed-precision evaluation
- torch.compile testing
- latency and throughput benchmarking
- numerical validation

> Available for GPU performance audits and PyTorch/CUDA optimization projects.

[Hire me on Fiverr](https://www.fiverr.com/s/r3ExR5y)
## GPUOpt Benchmarks

- **ResNet-18:** 3.22x speedup on NVIDIA Tesla T4
- **DistilBERT:** 5.28x speedup and 81.05% lower median latency on NVIDIA Tesla T4
# GPUOpt v1.0 — ResNet-18 GPU Optimization Case Study

GPUOpt is a PyTorch GPU optimization prototype that analyzes inference workloads,
evaluates execution strategies, validates numerical correctness, and promotes
optimizations only when they provide a measured performance benefit.

# GPUOpt v1.0 - ResNet-18 Performance Case Study

GPUOpt is a PyTorch GPU optimization prototype that analyzes inference workloads, evaluates execution policies, validates numerical correctness, and promotes a candidate only when it passes performance and safety checks.

## Case study

**Workload:** Pretrained ResNet-18 inference  
**GPU:** NVIDIA Tesla T4 (compute capability 7.5)  
**PyTorch:** 2.10.0+cu128  
**CUDA:** 12.8  
**Batch size:** 32  
**Input shape:** `(32, 3, 224, 224)`

| Metric | Original PyTorch | GPUOpt selected |
|---|---:|---:|
| Median latency | 27.240 ms | 8.453 ms |
| Throughput | 1174.8 images/s | 3785.7 images/s |
| Precision | FP32 | FP16 |
| Policy | ORIGINAL_FP32 | COMPILED_FP16 |

**Measured speedup:** **3.22x**  
**Latency reduction:** **68.97%**  
**Decision:** **CONFIRMED_IMPROVEMENT**

## Numerical validation

- RMSE: `0.00553673`
- Relative RMSE: `0.00403853`
- Mean absolute error: `0.00448727`
- Maximum absolute error: `0.01915741`
- Top-1 output agreement vs FP32 baseline on the tested batch: **100%**

## Benchmark methodology

- Same GPU session for baseline and selected policy
- CUDA-event timing
- Warm-up before measurement
- Randomized interleaved baseline/optimized trial order
- 20 warm-up iterations
- 50 repeats per trial
- 15 trials
- 3% minimum improvement threshold for confirmed promotion
- Original model preserved

## Interpretation

For this exact tested workload, GPUOpt reduced median latency from **27.24 ms** to **8.45 ms** and increased throughput from **1175** to **3786 images/s**.

The measured **68.97% execution-time reduction** should not be interpreted as a guaranteed cloud-cost reduction. Real production savings depend on utilization, batching, serving architecture, memory pressure, and infrastructure pricing.

## Current v1.0 scope

GPUOpt v1.0 is focused on PyTorch inference optimization and safe policy selection. It is not positioned as a universal replacement for TensorRT, Triton, or mature production compilers.
## Full Case Study

[View the one-page GPUOpt case study](case-study/GPUOpt_v1.0_ResNet18_Case_Study.pdf)

## Case Study #2 — DistilBERT Transformer Optimization

GPUOpt was evaluated on a real pretrained Transformer workload:

- **Model:** `distilbert-base-uncased-finetuned-sst-2-english`
- **GPU:** NVIDIA Tesla T4
- **Batch size:** 16
- **Sequence length:** 128
- **Selected fast policy:** `COMPILED_FP16`

### Results

- **Baseline median latency:** 64.02 ms
- **GPUOpt median latency:** 12.13 ms
- **Speedup:** 5.28x
- **Latency reduction:** 81.05%
- **Baseline throughput:** 249.9 samples/s
- **GPUOpt throughput:** 1,318.6 samples/s
- **Top-1 agreement:** 100% on the tested batch
- **Decision:** `CONFIRMED_IMPROVEMENT`

### Validation

The optimized policy preserved strong numerical agreement with the baseline:

- RMSE: 0.000974
- Relative RMSE: 0.000236
- Mean absolute error: 0.000805
- Max absolute error: 0.001903

> These results are workload-specific and were measured in the same GPU session using CUDA-event timing and randomized interleaved trials.
