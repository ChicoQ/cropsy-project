# A simple image classification project

## ENV set-up

### 2 GPUs

GTX1080 and GTX1080i

### Creating a new conda environment

  - conda update -n base -c defaults conda
  - conda create --name mmlab python=3.8 -y
  - conda activate mmlab
  - conda install pytorch torchvision cudatoolkit=11.0 -c pytorch

### Installing mmclassification

  - git clone https://github.com/open-mmlab/mmclassification
  - fowlling the instruction to install


## Splitting train/val from the original dataset

  - run `python moveFile.py` 3 times
  - the original dataset is splitted by 70:20:10 to train/val/test 
  - those subsets are moved under formatted_dataset/


## Labelling as ImageNet format

  - installing jupyter and kernel
  - as described in `labelling.ipynb`


## Training results

### training cli

```
(mmlab) chico@chico-nv:/hd1/cropsy/mmclassification$ ./tools/dist_train.sh configs/mobilenet_v2/mobilenet_v2_b32x8_cropsy.py 2
```

### no pipeline declared

2022-03-09 21:31:23,624 - mmcls - INFO - Saving checkpoint at 1 epochs
[>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>] 90/90, 38.8 task/s, elapsed: 2s, ETA:     0s

2022-03-09 21:31:26,055 - mmcls - INFO - Epoch(val) [1][2]	accuracy_top-1: 98.8889
2022-03-09 21:31:29,062 - mmcls - INFO - Saving checkpoint at 2 epochs
[>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>] 90/90, 38.8 task/s, elapsed: 2s, ETA:     0s

2022-03-09 21:31:31,494 - mmcls - INFO - Epoch(val) [2][2]	accuracy_top-1: 98.8889
2022-03-09 21:31:34,482 - mmcls - INFO - Saving checkpoint at 3 epochs
[>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>] 90/90, 38.6 task/s, elapsed: 2s, ETA:     0s

2022-03-09 21:31:36,923 - mmcls - INFO - Epoch(val) [3][2]	accuracy_top-1: 98.8889
2022-03-09 21:31:39,896 - mmcls - INFO - Saving checkpoint at 4 epochs
[>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>] 90/90, 39.0 task/s, elapsed: 2s, ETA:     0s

2022-03-09 21:31:42,433 - mmcls - INFO - Epoch(val) [4][2]	accuracy_top-1: 100.0000
2022-03-09 21:31:45,416 - mmcls - INFO - Saving checkpoint at 5 epochs
[>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>] 90/90, 38.9 task/s, elapsed: 2s, ETA:     0s

2022-03-09 21:31:47,846 - mmcls - INFO - Epoch(val) [5][2]	accuracy_top-1: 100.0000

### with pipelines

2022-03-09 21:37:34,668 - mmcls - INFO - Saving checkpoint at 1 epochs
[>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>] 90/90, 38.9 task/s, elapsed: 2s, ETA:     0s

2022-03-09 21:37:37,067 - mmcls - INFO - Epoch(val) [1][2]	accuracy_top-1: 97.7778
2022-03-09 21:37:40,039 - mmcls - INFO - Saving checkpoint at 2 epochs
[>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>] 90/90, 39.0 task/s, elapsed: 2s, ETA:     0s

2022-03-09 21:37:42,450 - mmcls - INFO - Epoch(val) [2][2]	accuracy_top-1: 98.8889
2022-03-09 21:37:45,447 - mmcls - INFO - Saving checkpoint at 3 epochs
[>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>] 90/90, 38.8 task/s, elapsed: 2s, ETA:     0s

2022-03-09 21:37:47,883 - mmcls - INFO - Epoch(val) [3][2]	accuracy_top-1: 100.0000
2022-03-09 21:37:50,873 - mmcls - INFO - Saving checkpoint at 4 epochs
[>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>] 90/90, 38.9 task/s, elapsed: 2s, ETA:     0s

2022-03-09 21:37:53,395 - mmcls - INFO - Epoch(val) [4][2]	accuracy_top-1: 100.0000
2022-03-09 21:37:56,348 - mmcls - INFO - Saving checkpoint at 5 epochs
[>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>] 90/90, 38.6 task/s, elapsed: 2s, ETA:     0s

2022-03-09 21:37:58,795 - mmcls - INFO - Epoch(val) [5][2]	accuracy_top-1: 100.0000

### no pipeline, but smaller lr 0.01 -> 0.005

optimizer = dict(type='SGD', lr=0.005, momentum=0.9, weight_decay=0.0001)

2022-03-09 21:42:12,439 - mmcls - INFO - Saving checkpoint at 1 epochs
[>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>] 90/90, 38.7 task/s, elapsed: 2s, ETA:     0s

2022-03-09 21:42:14,872 - mmcls - INFO - Epoch(val) [1][2]	accuracy_top-1: 93.3333
2022-03-09 21:42:17,838 - mmcls - INFO - Saving checkpoint at 2 epochs
[>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>] 90/90, 38.7 task/s, elapsed: 2s, ETA:     0s

2022-03-09 21:42:20,274 - mmcls - INFO - Epoch(val) [2][2]	accuracy_top-1: 97.7778
2022-03-09 21:42:23,231 - mmcls - INFO - Saving checkpoint at 3 epochs
[>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>] 90/90, 38.8 task/s, elapsed: 2s, ETA:     0s

2022-03-09 21:42:25,661 - mmcls - INFO - Epoch(val) [3][2]	accuracy_top-1: 98.8889
2022-03-09 21:42:28,620 - mmcls - INFO - Saving checkpoint at 4 epochs
[>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>] 90/90, 38.7 task/s, elapsed: 2s, ETA:     0s

2022-03-09 21:42:31,066 - mmcls - INFO - Epoch(val) [4][2]	accuracy_top-1: 100.0000
2022-03-09 21:42:34,004 - mmcls - INFO - Saving checkpoint at 5 epochs
[>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>] 90/90, 39.0 task/s, elapsed: 2s, ETA:     0s

2022-03-09 21:42:36,450 - mmcls - INFO - Epoch(val) [5][2]	accuracy_top-1: 100.0000

### samples_per_gpu 32 -> 64

2022-03-09 21:45:27,059 - mmcls - INFO - Saving checkpoint at 1 epochs
[>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>] 90/90, 37.6 task/s, elapsed: 2s, ETA:     0s

2022-03-09 21:45:29,572 - mmcls - INFO - Epoch(val) [1][1]	accuracy_top-1: 96.6667
2022-03-09 21:45:32,743 - mmcls - INFO - Saving checkpoint at 2 epochs
[>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>] 90/90, 37.6 task/s, elapsed: 2s, ETA:     0s

2022-03-09 21:45:35,246 - mmcls - INFO - Epoch(val) [2][1]	accuracy_top-1: 97.7778
2022-03-09 21:45:38,456 - mmcls - INFO - Saving checkpoint at 3 epochs
[>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>] 90/90, 37.6 task/s, elapsed: 2s, ETA:     0s

2022-03-09 21:45:40,960 - mmcls - INFO - Epoch(val) [3][1]	accuracy_top-1: 98.8889
2022-03-09 21:45:44,148 - mmcls - INFO - Saving checkpoint at 4 epochs
[>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>] 90/90, 37.5 task/s, elapsed: 2s, ETA:     0s

2022-03-09 21:45:46,661 - mmcls - INFO - Epoch(val) [4][1]	accuracy_top-1: 98.8889
2022-03-09 21:45:49,820 - mmcls - INFO - Saving checkpoint at 5 epochs
[>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>] 90/90, 37.6 task/s, elapsed: 2s, ETA:     0s

2022-03-09 21:45:52,395 - mmcls - INFO - Epoch(val) [5][1]	accuracy_top-1: 98.8889

### epocs 5 -> 8

2022-03-09 21:47:15,349 - mmcls - INFO - Saving checkpoint at 1 epochs
[>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>] 90/90, 37.5 task/s, elapsed: 2s, ETA:     0s

2022-03-09 21:47:17,865 - mmcls - INFO - Epoch(val) [1][1]	accuracy_top-1: 91.1111
2022-03-09 21:47:21,039 - mmcls - INFO - Saving checkpoint at 2 epochs
[>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>] 90/90, 37.5 task/s, elapsed: 2s, ETA:     0s

2022-03-09 21:47:23,559 - mmcls - INFO - Epoch(val) [2][1]	accuracy_top-1: 96.6667
2022-03-09 21:47:26,736 - mmcls - INFO - Saving checkpoint at 3 epochs
[>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>] 90/90, 37.5 task/s, elapsed: 2s, ETA:     0s

2022-03-09 21:47:29,244 - mmcls - INFO - Epoch(val) [3][1]	accuracy_top-1: 96.6667
2022-03-09 21:47:32,421 - mmcls - INFO - Saving checkpoint at 4 epochs
[>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>] 90/90, 37.5 task/s, elapsed: 2s, ETA:     0s

2022-03-09 21:47:34,934 - mmcls - INFO - Epoch(val) [4][1]	accuracy_top-1: 97.7778
2022-03-09 21:47:38,115 - mmcls - INFO - Saving checkpoint at 5 epochs
[>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>] 90/90, 37.5 task/s, elapsed: 2s, ETA:     0s

2022-03-09 21:47:40,625 - mmcls - INFO - Epoch(val) [5][1]	accuracy_top-1: 97.7778
2022-03-09 21:47:43,800 - mmcls - INFO - Saving checkpoint at 6 epochs
[>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>] 90/90, 37.5 task/s, elapsed: 2s, ETA:     0s

2022-03-09 21:47:46,312 - mmcls - INFO - Epoch(val) [6][1]	accuracy_top-1: 98.8889
2022-03-09 21:47:49,467 - mmcls - INFO - Saving checkpoint at 7 epochs
[>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>] 90/90, 37.7 task/s, elapsed: 2s, ETA:     0s

2022-03-09 21:47:51,968 - mmcls - INFO - Epoch(val) [7][1]	accuracy_top-1: 100.0000
2022-03-09 21:47:55,137 - mmcls - INFO - Saving checkpoint at 8 epochs
[>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>] 90/90, 37.5 task/s, elapsed: 2s, ETA:     0s

2022-03-09 21:47:57,644 - mmcls - INFO - Epoch(val) [8][1]	accuracy_top-1: 100.0000

### topk(1,2,3)

2022-03-09 21:56:03,197 - mmcls - INFO - Saving checkpoint at 1 epochs
[>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>] 90/90, 37.6 task/s, elapsed: 2s, ETA:     0s

2022-03-09 21:56:05,751 - mmcls - INFO - Epoch(val) [1][1]	accuracy_top-1: 82.2222, accuracy_top-2: 98.8889, accuracy_top-3: 100.0000
2022-03-09 21:56:08,915 - mmcls - INFO - Saving checkpoint at 2 epochs
[>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>] 90/90, 37.5 task/s, elapsed: 2s, ETA:     0s

2022-03-09 21:56:11,440 - mmcls - INFO - Epoch(val) [2][1]	accuracy_top-1: 95.5556, accuracy_top-2: 98.8889, accuracy_top-3: 100.0000
2022-03-09 21:56:14,610 - mmcls - INFO - Saving checkpoint at 3 epochs
[>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>] 90/90, 37.5 task/s, elapsed: 2s, ETA:     0s

2022-03-09 21:56:17,112 - mmcls - INFO - Epoch(val) [3][1]	accuracy_top-1: 96.6667, accuracy_top-2: 100.0000, accuracy_top-3: 100.0000
2022-03-09 21:56:20,263 - mmcls - INFO - Saving checkpoint at 4 epochs
[>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>] 90/90, 37.6 task/s, elapsed: 2s, ETA:     0s

2022-03-09 21:56:22,775 - mmcls - INFO - Epoch(val) [4][1]	accuracy_top-1: 97.7778, accuracy_top-2: 100.0000, accuracy_top-3: 100.0000
2022-03-09 21:56:25,936 - mmcls - INFO - Saving checkpoint at 5 epochs
[>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>] 90/90, 37.4 task/s, elapsed: 2s, ETA:     0s

2022-03-09 21:56:28,449 - mmcls - INFO - Epoch(val) [5][1]	accuracy_top-1: 98.8889, accuracy_top-2: 100.0000, accuracy_top-3: 100.0000
2022-03-09 21:56:31,603 - mmcls - INFO - Saving checkpoint at 6 epochs
[>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>] 90/90, 37.4 task/s, elapsed: 2s, ETA:     0s

2022-03-09 21:56:34,123 - mmcls - INFO - Epoch(val) [6][1]	accuracy_top-1: 98.8889, accuracy_top-2: 100.0000, accuracy_top-3: 100.0000
2022-03-09 21:56:37,282 - mmcls - INFO - Saving checkpoint at 7 epochs
[>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>] 90/90, 37.5 task/s, elapsed: 2s, ETA:     0s

2022-03-09 21:56:39,797 - mmcls - INFO - Epoch(val) [7][1]	accuracy_top-1: 100.0000, accuracy_top-2: 100.0000, accuracy_top-3: 100.0000
2022-03-09 21:56:42,943 - mmcls - INFO - Saving checkpoint at 8 epochs
[>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>] 90/90, 37.6 task/s, elapsed: 2s, ETA:     0s

2022-03-09 21:56:45,448 - mmcls - INFO - Epoch(val) [8][1]	accuracy_top-1: 100.0000, accuracy_top-2: 100.0000, accuracy_top-3: 100.0000

## Conclusion

Accoring to the above results, that `bunch` class still needs to improve the accuracy but the others look better.
