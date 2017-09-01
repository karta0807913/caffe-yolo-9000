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

  some image
> http://imgur.com/a/mXhat

```Shell
  cd build/example/yolo_9000_test/
  ./netforward ../../../examples/yolo/darknet_deploy.prototxt path/to/voc_model_iter_30000.caffemode path/to/voc_train.binaryproto test_list > output.txt
```
  test list is a txt file llike
```Shell
  /VOCdevkit/VOC2007/JPEGImages/000001.jpg
  /VOCdevkit/VOC2007/JPEGImages/000002.jpg
  /VOCdevkit/VOC2007/JPEGImages/000003.jpg
  ...
```
  it will output <label,x,y,w,h> per image 
  

## TODO 
  world tree 
  reorg layer
  Image Net Pretrain 
