#include "feature.h"
#include "svm.h"
#include <tiny_cnn\tiny_cnn.h>
using namespace tiny_cnn;
using namespace tiny_cnn::activation;


std::vector<feature_t> get_features_from_images(const std::vector<cv::Mat> &images, const p_static_feature_function func);
std::vector<feature_t> get_features_from_images(const std::vector<cv::Mat> &images, const p_static_feature_function_with_delta func, const int delta);

//svm model
svm_problem * svm_fill_problem(const std::vector<feature_t> &feats, const std::vector<int> &labels);
void svm_free_problem(svm_problem *prob);
svm_parameter * svm_fill_parameter(const double gamma = 0.5, const int C = 1000);
double svm_test_acc(const svm_problem * prob, const svm_model * model);
void svm_train_and_test(const std::vector<feature_t> &feats, const std::vector<int> &labels, const char * model_file, std::ofstream &ofs);
void svm_train_and_test(const std::vector<feature_t> &feats, const std::vector<int> &labels, const char * model_file);
double svm_test_acc(const std::vector<feature_t> &feats, const std::vector<int> &labels, const svm_model * model);
double svm_train_and_test(const std::vector<feature_t> &train_feats, const std::vector<int> &train_labels, const std::vector<feature_t> &test_feats, const std::vector<int> &test_labels, const char * model_file);
