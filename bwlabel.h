/*
author : wangzijie
date   : 2014/11/1
description:
c implement for bwlabel which is in matlab
parameters:
  in    --  a binary matrix
  num   --  capture the number of areas
  mode  --  8 or  4
return:
  matrix contains areas
*/

#pragma once
#include <opencv2\core\core.hpp>
cv::Mat bwlabel(cv::Mat in, int & num, int mode);
