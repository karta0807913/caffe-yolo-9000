CAFFE_HOME=../..

$CAFFE_HOME/build/tools/convert_box_data --label_file=map.txt --resize_height=416 --resize_width=416 --shuffle=true $CAFFE_HOME/ \
 $CAFFE_HOME/trainval.txt lmdb/yolo9000_train_lmdb
$CAFFE_HOME/build/tools/convert_box_data --label_file=map.txt --resize_height=416 --resize_width=416 --shuffle=true $CAFFE_HOME/ \
 $CAFFE_HOME/test_2007.txt lmdb/yolo9000_test_lmdb

$CAFFE_HOME/build/tools/compute_image_mean lmdb/yolo9000_train_lmdb lmdb/yolo9000_train.binaryproto
$CAFFE_HOME/build/tools/compute_image_mean lmdb/yolo9000_test_lmdb  lmdb/yolo9000_test.binaryproto
