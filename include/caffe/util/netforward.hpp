#ifndef NETFORWARD_H
#define NETFORWARD_H

#include "caffe/caffe.hpp"
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <algorithm>
#include <iosfwd>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include <opencv2/opencv.hpp>

class NetForward
{
public:
    NetForward(std::string proto, std::string model);
    NetForward(std::string proto, std::string model, std::string mean);
    NetForward(std::string proto, std::string model, caffe::Phase phase);
    NetForward(std::string proto, std::string model, std::string mean, caffe::Phase phase);
    std::vector<float> forward(cv::Mat &img);
    void printLayers();
    ~NetForward();

private:
    cv::Mat SetMean(const std::string& mean_file);
    void WrapInputLayer(std::vector<cv::Mat>* input_channels);
    void Preprocess(const cv::Mat& img, std::vector<cv::Mat>* input_channels);

    cv::Mat mean;
    caffe::Net<float> *net;
    caffe::Blob<float>* input_layer;
    caffe::NetParameter *param;
};
#endif // NETFORWARD_H
