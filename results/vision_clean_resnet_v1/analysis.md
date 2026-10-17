# Clean ResNet50 screen analysis

Frozen manifest: `97414fc5d2fac35463eadc88b88473073a35ac22b31d1f82eb595a36fd8e7ee0`; seed 42; test images 19,923.

## Selected checkpoints

| Stage | Zero-based epoch | Bytes | SHA-256 |
|---|---:|---:|---|
| BASE | 5 | 284,147,575 | `bd99ba9d844671b86f7b887cbfe1da4a3136cf9b178ba8f2f042f6df4c558f89` |
| CE | 3 | 284,145,825 | `75f16b839bff9e84c13189283d7cc0e3091dbcbfae0a097c6d7446c4504375d5` |
| F2 | 3 | 284,145,825 | `bfa53066ee7b69cc6d34637c9df9d585968228b938765cb5e9719636b4f0c854` |

## Paired comparisons

| Comparison | Difference (pp) | Group-bootstrap 95% CI (pp) | Left wins | Right wins |
|---|---:|---:|---:|---:|
| CE minus BASE | 0.241 | [0.025, 0.444] | 227 | 179 |
| F2 minus BASE | -0.146 | [-0.413, 0.100] | 251 | 280 |
| F2 minus CE | -0.386 | [-0.614, -0.160] | 182 | 259 |

| Source | Test n | Base (%) | CE (%) | F2 (%) | F2-CE (pp) |
|---|---:|---:|---:|---:|---:|
| ayushbpanchal/cashew-image-dataset | 3,877 | 96.47 | 96.67 | 96.03 | -0.64 |
| binnassor89/melon-macronutrient-deficiency-dataset | 29 | 100.00 | 100.00 | 100.00 | 0.00 |
| mexwell/crop-diseases-classification | 2,691 | 80.01 | 81.12 | 80.97 | -0.15 |
| nirmalsankalana/rocole-a-robusta-coffee-leaf-images-dataset | 229 | 88.21 | 86.03 | 85.59 | -0.44 |
| shubham2703/five-crop-diseases-dataset | 1,978 | 95.35 | 95.40 | 94.34 | -1.06 |
| tahmidmir/pumpkin-leaf-diseases-dataset-from-bangladesh | 300 | 86.00 | 85.00 | 83.67 | -1.33 |
| tushar5harma/plant-village-dataset-updated | 10,062 | 98.79 | 98.94 | 98.80 | -0.14 |
| yasaminborhani/wheatrustclassificationdataset | 551 | 98.73 | 99.09 | 97.82 | -1.27 |
| zunorain/pea-plant-dataset | 206 | 96.60 | 96.60 | 96.12 | -0.49 |

## Post hoc error categories

| Checkpoint | Correct | Wrong condition, same crop | Wrong crop |
|---|---:|---:|---:|
| BASE | 18,951 | 888 | 84 |
| CE | 18,999 | 850 | 74 |
| F2 | 18,922 | 933 | 68 |

The primary CE-versus-F2 result is negative for F2. The interval represents resampling of test groups only and does not justify a multi-seed or external-domain claim. The error-category analysis was performed after the test read and is descriptive.
