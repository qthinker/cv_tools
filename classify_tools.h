#include "feature.h"
#include "svm.h"
typedef feature_t (*p_func_feature_extract)(cv::Mat &);
typedef feature_t (*p_func_feature_extract_with_delta)(cv::Mat &, int);

//template<typename F>
std::vector<feature_t> get_features_from_images(std::vector<cv::Mat> &images, const p_func_feature_extract func);
std::vector<feature_t> get_features_from_images(std::vector<cv::Mat> &images, const p_func_feature_extract_with_delta func, int delta);
svm_problem * fill_problem(std::vector<feature_t> feats, std::vector<int> labels);
void free_problem(svm_problem *prob);
svm_parameter * fill_parameter(double gamma = 0.5, int C = 1000);
void free_problem(svm_problem *prob);
double test_acc(svm_problem * prob, svm_model * model);
void train_and_test(std::vector<feature_t> feats, std::vector<int> labels, const char * model_file, std::ofstream &ofs);
void train_and_test(std::vector<feature_t> feats, std::vector<int> labels, const char * model_file);
double test_acc(std::vector<feature_t> feats, std::vector<int> labels, svm_model * model);
double train_and_test(std::vector<feature_t> train_feats, std::vector<int> train_labels,std::vector<feature_t> test_feats, std::vector<int> test_labels, const char * model_file);