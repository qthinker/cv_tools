#pragma once
#include <opencv2\core\core.hpp>
cv::Mat bwlabel(const cv::Mat in, int * num = NULL, const int mode = 8);