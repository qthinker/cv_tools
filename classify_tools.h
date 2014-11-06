#include "feature.h"
#include "svm.h"
#include <tiny_cnn\tiny_cnn.h>
using namespace tiny_cnn;
using namespace tiny_cnn::activation;
typedef feature_t (*p_func_feature_extract)(cv::Mat &);
typedef feature_t (*p_func_feature_extract_with_delta)(cv::Mat &, int);

//template<typename F>
std::vector<feature_t> get_features_from_images(std::vector<cv::Mat> &images, const p_func_feature_extract func);
std::vector<feature_t> get_features_from_images(std::vector<cv::Mat> &images, const p_func_feature_extract_with_delta func, int delta);
svm_problem * svm_fill_problem(std::vector<feature_t> feats, std::vector<int> labels);
void svm_free_problem(svm_problem *prob);
svm_parameter * svm_fill_parameter(double gamma = 0.5, int C = 1000);
void svm_free_problem(svm_problem *prob);
double svm_test_acc(svm_problem * prob, svm_model * model);
void svm_train_and_test(std::vector<feature_t> feats, std::vector<int> labels, const char * model_file, std::ofstream &ofs);
void svm_train_and_test(std::vector<feature_t> feats, std::vector<int> labels, const char * model_file);
double svm_test_acc(std::vector<feature_t> feats, std::vector<int> labels, svm_model * model);
double svm_train_and_test(std::vector<feature_t> train_feats, std::vector<int> train_labels,std::vector<feature_t> test_feats, std::vector<int> test_labels, const char * model_file);
void mlp_train(const char * positive_path, const char * negetive_path, const char * weight_file, p_func_feature_extract func);
tiny_cnn::result mlp_test(const char * positive_path, const char * negetive_path, const char * weight_file, p_func_feature_extract func);