#include <cstddef>
#include <vector>
#include <opencv2\opencv.hpp>

void load_image_blocks_from_path(const char * positive_path, const char * negative_path,
								 std::vector<cv::Mat> &images, std::vector<int> &labels,
								 int pos_label, int neg_label);

void train_test_split(std::vector<cv::Mat> & images, std::vector<int> & labels, 
					  std::vector<cv::Mat> & train_images, std::vector<int> & train_labels,
					  std::vector<cv::Mat> & test_images, std::vector<int> & test_labels, double test_size);