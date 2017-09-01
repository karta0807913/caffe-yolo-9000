#include "caffe/util/netforward.h"

using namespace caffe;
using namespace std;

NetForward::NetForward(std::string proto, std::string model)
{
    NetForward(proto, model, "", TEST);
}

NetForward::NetForward(std::string proto, std::string model, std::string mean)
{
    NetForward(proto, model, mean, TEST);
}

NetForward::NetForward(std::string proto, std::string model, caffe::Phase phase)
{
    NetForward(proto, model, "", phase);
}

NetForward::NetForward(std::string proto, std::string model, std::string mean,
                       caffe::Phase phase)
{
    Caffe::set_mode(Caffe::GPU);
    Caffe::SetDevice(1);
    net = new caffe::Net<float>(proto, phase);
    param = new NetParameter();
    /* init net */
    ReadNetParamsFromBinaryFileOrDie(model, param);
    net->CopyTrainedLayersFrom(model.c_str());

    input_layer = net->input_blobs()[0];

    if(mean != ""){
        this->mean = SetMean(mean);
    }
    else
    {
        this->mean = cv::Mat();
    }
}

vector<float> NetForward::forward(cv::Mat &image)
{
    input_layer->Reshape(1, input_layer->channels(),
                         input_layer->width(), input_layer->height());
    net->Reshape();

    std::vector<cv::Mat> input_channels;
    WrapInputLayer(&input_channels);
    Preprocess(image, &input_channels);
    net->Forward();

    Blob<float>* output_layer = net->output_blobs()[0];
    const float* begin = output_layer->cpu_data();
    const float* end = begin + output_layer->channels();

    //delete input_channels;

    return vector<float>(begin, end);
}

void NetForward::printLayers()
{
    int num_layers = param->layer_size();
    for (int i = 0; i < num_layers; ++i)
    {
        cout << "Layer " << i << ":" << param->layer(i).name()
             << "\t" << param->layer(i).type() << endl;
    }
}

cv::Mat NetForward::SetMean(const string& mean_file) {
  BlobProto blob_proto;
  ReadProtoFromBinaryFileOrDie(mean_file.c_str(), &blob_proto);

  /* Convert from BlobProto to Blob<float> */
  Blob<float> mean_blob;
  mean_blob.FromProto(blob_proto);
  CHECK_EQ(mean_blob.channels(), input_layer->channels())
    << "Number of channels of mean file doesn't match input layer.";

  /* The format of the mean file is planar 32-bit float BGR or grayscale. */
  std::vector<cv::Mat> channels;
  float* data = mean_blob.mutable_cpu_data();
  for (int i = 0; i < input_layer->channels(); ++i) {
    /* Extract an individual channel. */
    cv::Mat channel(mean_blob.height(), mean_blob.width(), CV_32FC1, data);
    channels.push_back(channel);
    data += mean_blob.height() * mean_blob.width();
  }

  /* Merge the separate channels into a single image. */
  cv::Mat mean;
  cv::merge(channels, mean);

  /* Compute the global mean pixel value and create a mean image
   * filled with this value. */
  cv::Scalar channel_mean = cv::mean(mean);
  return cv::Mat(cv::Size(input_layer->width(), input_layer->height()),
                 mean.type(), channel_mean);
}

void NetForward::WrapInputLayer(std::vector<cv::Mat>* input_channels) {

  int width = input_layer->width();
  int height = input_layer->height();
  float* input_data = input_layer->mutable_cpu_data();
  for (int i = 0; i < input_layer->channels(); ++i) {
    cv::Mat channel(height, width, CV_32FC1, input_data);
    input_channels->push_back(channel);
    input_data += width * height;
  }
}

void NetForward::Preprocess(const cv::Mat& img,
               std::vector<cv::Mat>* input_channels) {
  /* Convert the input image to the input image format of the network. */

  int num_channels_ = input_layer->channels();
  cv::Mat sample;
  if (img.channels() == 3 && num_channels_ == 1)
    cv::cvtColor(img, sample, cv::COLOR_BGR2GRAY);
  else if (img.channels() == 4 && num_channels_ == 1)
    cv::cvtColor(img, sample, cv::COLOR_BGRA2GRAY);
  else if (img.channels() == 4 && num_channels_ == 3)
    cv::cvtColor(img, sample, cv::COLOR_BGRA2BGR);
  else if (img.channels() == 1 && num_channels_ == 3)
    cv::cvtColor(img, sample, cv::COLOR_GRAY2BGR);
  else
    //cv::cvtColor(img, sample, cv::COLOR_RGB2BGR);
    sample = img;

  cv::Mat sample_resized;
  if (sample.size() != cv::Size(input_layer->width(), input_layer->height()))
    cv::resize(sample, sample_resized, cv::Size(input_layer->width(), input_layer->height()));
  else
    sample_resized = sample;

  cv::Mat sample_float;
  if (num_channels_ == 3)
    sample_resized.convertTo(sample_float, CV_32FC3);
  else
    sample_resized.convertTo(sample_float, CV_32FC1);

  cv::Mat sample_normalized;
  if(mean.empty()){
      sample_normalized = sample_float;
  }
  else{
      cv::subtract(sample_float, mean, sample_normalized);
  }

  /* This operation will write the separate BGR planes directly to the
   * input layer of the network because it is wrapped by the cv::Mat
   * objects in input_channels. */

  cv::split(sample_normalized, *input_channels);
}

NetForward::~NetForward()
{
    delete param;
    delete net;
}

