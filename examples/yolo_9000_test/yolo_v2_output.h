#ifndef YOLO_V2_OUTPUT_H
#define YOLO_V2_OUTPUT_H

#include <vector>
#include <algorithm>
#include <iostream>

namespace yolo_v2{

    class DATA{
    public:
        float confidence;

        float   x,
                y,
                w,
                h;

        int classIndex;

        bool operator<(DATA data)
        {
            return this->confidence > data.confidence;
        }

    private:
        bool operator>(DATA data);
    };

    float sigmoid(double x)
    {
        return 1 / (1 + exp(-x));
    }

    template<typename Dtype>
    Dtype Overlap(Dtype x1, Dtype w1, Dtype x2, Dtype w2) {
      Dtype left = std::max(x1, x2);
      Dtype right = std::min(x1 + w1, x2 + w2);
      return right - left;
    }

    float getTwoBoxArea(DATA data1, DATA data2)
    {
        float width = Overlap(data1.x, data1.w,
                          data2.x, data1.w);

        float height = Overlap(data1.y, data1.h,
                          data2.y, data1.h);

        if (width <= 0 || height <= 0)
            return 0;
        else
        {
            float area = width * height;
            float area1 = data1.w * data1.h;
            float area2 = data2.w * data2.h;
            return area / std::min(area1, area2);
        }
    }

    template<int OBJECT_NUM_, int SIDE_>
    std::vector<DATA> getResult(std::vector<float> &res, std::vector<float> &bias, int classNumber,
                              float confidenceLimit, float twoBoxArea){
        if((int)res.size() != (OBJECT_NUM_) * (classNumber + 5) * SIDE_ * SIDE_){
            std::cout << "vector Number (" << res.size() << ") != request Number (" << OBJECT_NUM_ * (classNumber + 5) * SIDE_ * SIDE_ << ")" << std::endl;
            exit(1);
        }
        if(OBJECT_NUM_ > bias.size() / 2){
            std::cout << "OBJECT NUMBER (" << OBJECT_NUM_ << ") > bias number / 2 (" << bias.size() / 2 << ")" << std::endl;
            exit(1);
        }
        DATA *data = new DATA[OBJECT_NUM_];
        std::vector<DATA> ans;
        const int screen_size = SIDE_ * SIDE_;
        const int box_size = (5 + classNumber) * screen_size;

        for(int side_y = 0; side_y < SIDE_; ++side_y){
            for(int side_x = 0; side_x < SIDE_; ++side_x){
                for(int obj_index = 0; obj_index < OBJECT_NUM_; ++obj_index){
                    int box_index = side_y * SIDE_ + side_x + obj_index * box_size;
                    data[obj_index].x = (sigmoid(res[box_index + 0 * screen_size]) + side_x) / SIDE_;
                    data[obj_index].y = (sigmoid(res[box_index + 1 * screen_size]) + side_y) / SIDE_;
                    data[obj_index].w = bias[2 * obj_index + 0] * exp(res[box_index + 2 * screen_size]);
                    data[obj_index].h = bias[2 * obj_index + 1] * exp(res[box_index + 3 * screen_size]);
                    data[obj_index].confidence = res[box_index + 4 * screen_size];

                    int pred_class = 0;
                    int best_score = 0;
                    for(int class_index = 0; class_index < classNumber; ++class_index){
                        if(res[box_index + (5 + class_index) * screen_size] > best_score){
                            best_score = res[box_index + (5 + class_index) * screen_size];
                            pred_class = class_index;
                        }
                    }
                    data[obj_index].classIndex = pred_class;
                }
                std::sort(data, data + OBJECT_NUM_);
                bool flag[OBJECT_NUM_] = { false };
                for(int col = 0; col < OBJECT_NUM_; ++col)
                {
                    if(data[col].confidence < confidenceLimit)
                        break;
                    if(flag[col])
                        continue;
                    for(int raw = col + 1; raw < OBJECT_NUM_; ++raw)
                    {
                        if(flag[raw])
                            continue;
                        if(data[raw].confidence < confidenceLimit)
                        {
                            break;
                        }
                        float iou = getTwoBoxArea(data[col], data[raw]);
                        if(iou > twoBoxArea)
                        {
                            flag[raw] = true;
                            continue;
                        }
                    }
                    ans.push_back(data[col]);
                }
            }
        }
        delete data;
        return ans;
    }
}
#else
namespace yolo_v2 {

}
#endif // YOLO_V2_OUTPUT_H
