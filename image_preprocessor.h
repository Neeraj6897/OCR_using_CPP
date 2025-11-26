#pragma once

#include <vector>
#include <string>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

class ImagePreprocessor {
public:
    static vector<float> preprocessImageForMNIST(const string& image_path);
    
    static Mat loadImage(const string& image_path);
    static Mat convertToGrayscale(const Mat& image);
    static Mat enhanceContrast(const Mat& image);
    static Mat removeNoise(const Mat& image);
    static Mat centerDigit(const Mat& image);
    static Mat resizeToMNIST(const Mat& image);
    static Mat invertColors(const Mat& image);
    static vector<float> matToFloat(const Mat& image);
    
    static void saveDebugImage(const Mat& image, const string& output_path);
    static bool isImageFile(const string& filename);
};