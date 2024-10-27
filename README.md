# benchmark resize [Pillow, resizer, cvcuda]

This is a benchmark to compare the speed of resizing with Pillow, resizer, and cvcuda.

## Environment

```bash
 uv run python -m torch.utils.collect_env
```
  
```markdown

<frozen runpy>:128: RuntimeWarning: 'torch.utils.collect_env' found in sys.modules after import of package 'torch.utils', but prior to execution of 'torch.utils.collect_env'; this may result in unpredictable behaviour
Collecting environment information...
PyTorch version: 2.5.0+cu124
Is debug build: False
CUDA used to build PyTorch: 12.4
ROCM used to build PyTorch: N/A

OS: Debian GNU/Linux 11 (bullseye) (x86_64)
GCC version: (Debian 10.2.1-6) 10.2.1 20210110
Clang version: Could not collect
CMake version: version 3.30.5
Libc version: glibc-2.31

Python version: 3.11.8 (main, Feb 25 2024, 04:18:18) [Clang 17.0.6 ] (64-bit runtime)
Python platform: Linux-5.10.0-33-cloud-amd64-x86_64-with-glibc2.31
Is CUDA available: True
CUDA runtime version: 11.8.89
CUDA_MODULE_LOADING set to: LAZY
GPU models and configuration: GPU 0: NVIDIA L4
Nvidia driver version: 550.90.07
cuDNN version: Could not collect
HIP runtime version: N/A
MIOpen runtime version: N/A
Is XNNPACK available: True

CPU:
Architecture:                         x86_64
CPU op-mode(s):                       32-bit, 64-bit
Byte Order:                           Little Endian
Address sizes:                        46 bits physical, 48 bits virtual
CPU(s):                               4
On-line CPU(s) list:                  0-3
Thread(s) per core:                   2
Core(s) per socket:                   2
Socket(s):                            1
NUMA node(s):                         1
Vendor ID:                            GenuineIntel
CPU family:                           6
Model:                                85
Model name:                           Intel(R) Xeon(R) CPU @ 2.20GHz
Stepping:                             7
CPU MHz:                              2200.200
BogoMIPS:                             4400.40
Hypervisor vendor:                    KVM
Virtualization type:                  full
L1d cache:                            64 KiB
L1i cache:                            64 KiB
L2 cache:                             2 MiB
L3 cache:                             38.5 MiB
NUMA node0 CPU(s):                    0-3
Vulnerability Gather data sampling:   Not affected
Vulnerability Itlb multihit:          Not affected
Vulnerability L1tf:                   Not affected
Vulnerability Mds:                    Not affected
Vulnerability Meltdown:               Not affected
Vulnerability Mmio stale data:        Vulnerable: Clear CPU buffers attempted, no microcode; SMT Host state unknown
Vulnerability Reg file data sampling: Not affected
Vulnerability Retbleed:               Mitigation; Enhanced IBRS
Vulnerability Spec rstack overflow:   Not affected
Vulnerability Spec store bypass:      Mitigation; Speculative Store Bypass disabled via prctl and seccomp
Vulnerability Spectre v1:             Mitigation; usercopy/swapgs barriers and __user pointer sanitization
Vulnerability Spectre v2:             Mitigation; Enhanced / Automatic IBRS, IBPB conditional, RSB filling, PBRSB-eIBRS SW sequence
Vulnerability Srbds:                  Not affected
Vulnerability Tsx async abort:        Vulnerable: Clear CPU buffers attempted, no microcode; SMT Host state unknown
Flags:                                fpu vme de pse tsc msr pae mce cx8 apic sep mtrr pge mca cmov pat pse36 clflush mmx fxsr sse sse2 ss ht syscall nx pdpe1gb rdtscp lm constant_tsc rep_good nopl xtopology nonstop_tsc cpuid tsc_known_freq pni pclmulqdq ssse3 fma cx16 pcid sse4_1 sse4_2 x2apic movbe popcnt aes xsave avx f16c rdrand hypervisor lahf_lm abm 3dnowprefetch invpcid_single ssbd ibrs ibpb stibp ibrs_enhanced fsgsbase tsc_adjust bmi1 hle avx2 smep bmi2 erms invpcid rtm mpx avx512f avx512dq rdseed adx smap clflushopt clwb avx512cd avx512bw avx512vl xsaveopt xsavec xgetbv1 xsaves arat avx512_vnni md_clear arch_capabilities

Versions of relevant libraries:
[pip3] numpy==2.1.2
[pip3] torch==2.5.0
[pip3] triton==3.1.0
[conda] Could not collect
```

## setup

```bash
uv sync
```

## Run benchmark

```bash
uv run pytest -s test_benchmark.py
```

## Result benchmark

```markdown
| Package (time in ms)       |   linear |   lanczos3 |   nearest |   bilinear |
|:---------------------------|---------:|-----------:|----------:|-----------:|
| cvcuda tensor              |     0.20 |       0.20 |           |            |
| cvcuda image               |    63.30 |      82.52 |           |            |
| cykooz.resizer             |          |     129.97 |      0.61 |      66.19 |
| cykooz.resizer - sse4_1    |          |      59.76 |      0.60 |      32.02 |
| cykooz.resizer - avx2      |          |      39.11 |      0.59 |      22.27 |
| Pillow U8                  |          |     110.90 |      0.37 |      36.86 |
| cykooz.resizer U8          |          |      29.10 |      0.32 |      13.76 |
| cykooz.resizer U8 - sse4_1 |          |      14.31 |      0.32 |       5.52 |
| cykooz.resizer U8 - avx2   |          |      10.95 |      0.31 |       5.18 |
| cvcuda tensor u8           |     0.18 |       0.18 |           |            |


========================================================================================================================== warnings summary ==========================================================================================================================


---------------------------------------------------------------------------------------------------------------- benchmark: 27 tests ----------------------------------------------------------------------------------------------------------------
Name (time in us)                                             Min                     Max                    Mean                 StdDev                  Median                    IQR            Outliers         OPS            Rounds  Iterations
-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
test_resize_pillow_cuda_from_tensor_u8[lanczos3]         173.6020 (1.0)          192.1580 (1.02)         179.7351 (1.0)           6.2501 (1.37)         178.2000 (1.0)           7.6561 (1.17)          2;0  5,563.7432 (1.0)          10           1
test_resize_pillow_cuda_from_tensor_u8[linear]           174.5070 (1.01)         187.8510 (1.0)          181.7212 (1.01)          4.5635 (1.0)          182.2715 (1.02)          7.5760 (1.15)          4;0  5,502.9353 (0.99)         10           1
test_resize_pillow_cuda_from_tensor[lanczos3]            188.5100 (1.09)         216.9610 (1.15)         202.3906 (1.13)          8.4848 (1.86)         203.6245 (1.14)          6.5600 (1.0)           3;2  4,940.9410 (0.89)         10           1
test_resize_pillow_cuda_from_tensor[linear]              190.0240 (1.09)         217.7470 (1.16)         201.4807 (1.12)         10.2350 (2.24)         198.1965 (1.11)         16.4100 (2.50)          3;0  4,963.2542 (0.89)         10           1
test_resize_pil_u8[nearest-sse4_1]                       305.1700 (1.76)         331.2740 (1.76)         315.1435 (1.75)          8.9028 (1.95)         313.2385 (1.76)         15.0120 (2.29)          5;0  3,173.1577 (0.57)         10           1
test_resize_pil_u8[nearest-avx2]                         305.3460 (1.76)         335.4510 (1.79)         314.2647 (1.75)          9.6423 (2.11)         311.7565 (1.75)         11.6270 (1.77)          1;0  3,182.0309 (0.57)         10           1
test_resize_pil_u8[nearest-none]                         306.3540 (1.76)         329.9180 (1.76)         315.1583 (1.75)          7.4469 (1.63)         315.8170 (1.77)         11.8100 (1.80)          3;0  3,173.0085 (0.57)         10           1
test_resize_pillow_u8[nearest]                           342.3130 (1.97)         462.2980 (2.46)         373.5955 (2.08)         25.9545 (5.69)         372.1065 (2.09)         30.2370 (4.61)          2;1  2,676.6917 (0.48)         20           1
test_resize_pil[nearest-avx2]                            571.8220 (3.29)         623.0310 (3.32)         592.3528 (3.30)         19.1698 (4.20)         586.4250 (3.29)         33.5600 (5.12)          3;0  1,688.1831 (0.30)         10           1
test_resize_pil[nearest-none]                            573.4670 (3.30)         666.8610 (3.55)         613.7530 (3.41)         32.3207 (7.08)         614.2825 (3.45)         63.5920 (9.69)          4;0  1,629.3199 (0.29)         10           1
test_resize_pil[nearest-sse4_1]                          577.2750 (3.33)         627.7571 (3.34)         601.5625 (3.35)         14.1852 (3.11)         598.7860 (3.36)         16.3680 (2.50)          3;0  1,662.3377 (0.30)         10           1
test_resize_pil_u8[bilinear-avx2]                      5,068.6960 (29.20)      5,235.8611 (27.87)      5,183.3686 (28.84)        55.7600 (12.22)      5,205.9475 (29.21)        79.4120 (12.11)         2;0    192.9247 (0.03)         10           1
test_resize_pil_u8[bilinear-sse4_1]                    5,458.7240 (31.44)      5,662.0630 (30.14)      5,515.0039 (30.68)        64.8785 (14.22)      5,494.8675 (30.84)        66.8520 (10.19)         2;1    181.3235 (0.03)         10           1
test_resize_pil_u8[lanczos3-avx2]                     10,822.3110 (62.34)     11,085.7960 (59.01)     10,951.2957 (60.93)       100.6304 (22.05)     11,008.9955 (61.78)       169.9650 (25.91)         4;0     91.3134 (0.02)         10           1
test_resize_pil_u8[bilinear-none]                     13,625.6840 (78.49)     13,980.6130 (74.42)     13,760.6970 (76.56)       109.0469 (23.90)     13,743.1150 (77.12)       133.0720 (20.29)         3;0     72.6707 (0.01)         10           1
test_resize_pil_u8[lanczos3-sse4_1]                   14,233.6680 (81.99)     14,402.0880 (76.67)     14,309.5769 (79.61)        62.2959 (13.65)     14,308.7715 (80.30)       117.5370 (17.92)         3;0     69.8833 (0.01)         10           1
test_resize_pil[bilinear-avx2]                        22,089.9410 (127.24)    22,445.1730 (119.48)    22,265.3275 (123.88)      122.6805 (26.88)     22,258.6520 (124.91)      169.7281 (25.87)         4;0     44.9129 (0.01)         10           1
test_resize_pil_u8[lanczos3-none]                     28,824.6590 (166.04)    29,534.0520 (157.22)    29,099.9586 (161.90)      226.9470 (49.73)     29,057.6185 (163.06)      396.1090 (60.38)         3;0     34.3643 (0.01)         10           1
test_resize_pil[bilinear-sse4_1]                      31,886.6240 (183.68)    32,202.6170 (171.43)    32,017.2611 (178.14)      106.0368 (23.24)     32,024.5760 (179.71)      120.6109 (18.39)         4;0     31.2332 (0.01)         10           1
test_resize_pillow_u8[bilinear]                       36,539.7750 (210.48)    37,342.5850 (198.79)    36,856.4690 (205.06)      164.7203 (36.10)     36,813.2995 (206.58)      163.1790 (24.87)         3;1     27.1323 (0.00)         20           1
test_resize_pil[lanczos3-avx2]                        38,816.1780 (223.59)    39,684.5700 (211.26)    39,105.1625 (217.57)      303.5146 (66.51)     38,987.6770 (218.79)      545.0870 (83.09)         2;0     25.5721 (0.00)         10           1
test_resize_pil[lanczos3-sse4_1]                      59,359.1760 (341.93)    60,751.9260 (323.40)    59,760.2506 (332.49)      437.0312 (95.77)     59,556.2115 (334.21)      541.5740 (82.56)         1;0     16.7335 (0.00)         10           1
test_resize_pillow_cuda_from_pil_image[lanczos3]      62,972.8130 (362.74)   102,222.9920 (544.17)    82,517.5321 (459.11)   20,432.0820 (>1000.0)   82,440.0220 (462.63)   38,770.0250 (>1000.0)       0;0     12.1186 (0.00)         10           1
test_resize_pillow_cuda_from_pil_image[linear]        63,065.8410 (363.28)    63,715.7960 (339.18)    63,295.8951 (352.16)      219.4429 (48.09)     63,260.5600 (355.00)      267.2580 (40.74)         3;0     15.7988 (0.00)         10           1
test_resize_pil[bilinear-none]                        65,483.8140 (377.21)    66,719.9630 (355.17)    66,188.0271 (368.25)      401.2182 (87.92)     66,202.7770 (371.51)      382.2331 (58.27)         4;0     15.1085 (0.00)         10           1
test_resize_pillow_u8[lanczos3]                      110,284.3650 (635.27)   113,495.0040 (604.18)   110,897.7991 (617.01)      682.1171 (149.47)   110,707.5655 (621.25)      265.5400 (40.48)         1;3      9.0173 (0.00)         20           1
test_resize_pil[lanczos3-none]                       129,275.4009 (744.67)   132,935.1900 (707.66)   129,974.7837 (723.15)    1,096.4253 (240.26)   129,550.4260 (726.99)      797.2610 (121.53)        1;1      7.6938 (0.00)         10           1
-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

```