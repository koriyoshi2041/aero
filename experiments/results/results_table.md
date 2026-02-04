# FF Attack Results

## Target Success Rate

| Strategy | Whitebox | vgg16_bn | mobilenetv2_x1_0 | shufflenetv2_x1_0 |
|---|---|---|---|---|
| random | 36.8% | 6.2% | 9.0% | 6.8% |
| least_likely | 27.4% | 3.8% | 3.4% | 3.2% |
| most_confusing | 60.2% | 13.8% | 20.0% | 16.4% |
| semantic | 53.8% | 9.4% | 13.8% | 11.8% |
| multi_target | 60.2% | 13.8% | 20.0% | 16.4% |
| dynamic_topk | 27.4% | 3.8% | 3.4% | 3.2% |