# CAFFE for YOLO-9000

## Reference

> YOLO9000: Better, Faster, Stronger

> https://arxiv.org/abs/1612.08242

> https://pjreddie.com/darknet/yolo/

> https://github.com/yeahkun/caffe-yolo

## Usage

### Data preparation
```Shell
  ln -s /your/path/to/VOCdevkit/ .
  python data/yolo/get_list.py
  cd data/yolo
  # change related path in script convert.sh
  ./convert.sh 
```

### Train
```Shell
  cd examples/yolo
  # change related path in script train.sh and darknet.prototxt
  ./train.sh
```

### Test

  weight
> https://drive.google.com/open?id=0Bx1yEotM0mQeM3ZPRmhFSHAydUE

  binaryproto
> https://drive.google.com/open?id=0Bx1yEotM0mQeTE9XYWJjazZTLUE

  Qt tool
> https://github.com/karta0807913/LabelingTools

## TODO 
  world tree 
  reorg layer
  Image Net Pretrain 
