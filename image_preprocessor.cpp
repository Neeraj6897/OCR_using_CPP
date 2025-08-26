#include "image_preprocessor.h"
#include <iostream>
#include <algorithm>

using namespace std;
using namespace cv;

vector<float> ImagePreprocessor::preprocessImageForMNIST(const string& image_path) {
    cout << "Processing image: " << image_path << endl;

    Mat original = loadImage(image_path);
    if (original.empty()) {
        throw runtime_error("Failed to load image: " + image_path);
    }
    Mat grayscale = convertToGrayscale(original);
    Mat centered = centerDigit(grayscale);
    Mat resized = resizeToMNIST(centered); //resize to 28x28

    vector<float> image_vector(784);
    for (int i = 0; i < resized.rows * resized.cols; ++i) {
        image_vector[i] = static_cast<float>(resized.at<unsigned char>(i)) / 255.0f; //normalize to get [0-1] values
    }

    imwrite("debug_processed.png", resized);
    cout << "Saved final preprocessed image to debug_processed.png" << endl;

    return image_vector;
}

Mat ImagePreprocessor::loadImage(const string& image_path) {
    
    Mat image = imread(image_path, IMREAD_UNCHANGED);
    
    if (image.empty()) {
        throw runtime_error("Cannot load image file: " + image_path);
    }
    
    return image;
}

Mat ImagePreprocessor::convertToGrayscale(const Mat& image) {
    Mat grayscale;
    
    if (image.channels() == 3) {
        cvtColor(image, grayscale, COLOR_BGR2GRAY);
    } else if (image.channels() == 4) {
        cvtColor(image, grayscale, COLOR_BGRA2GRAY);
    } else {
        grayscale = image.clone();
    }
    
    return grayscale;
}

Mat ImagePreprocessor::enhanceContrast(const Mat& image) {
    Mat enhanced;
    
    Ptr<CLAHE> clahe = createCLAHE();
    clahe->setClipLimit(2.0);
    clahe->apply(image, enhanced);
    
    return enhanced;
}

Mat ImagePreprocessor::removeNoise(const Mat& image) {
    Mat denoised;
    
    medianBlur(image, denoised, 3);
    
    return denoised;
}

Mat ImagePreprocessor::centerDigit(const Mat& image) {
    if (mean(image)[0] > 50.0) { // Heuristic: 50/255 is a good threshold for light backgrounds
        cout << "Detected light background, applying full centering logic." << endl;
        Mat binary;
        threshold(image, binary, 0, 255, THRESH_BINARY_INV + THRESH_OTSU);

        vector<vector<Point>> contours;
        findContours(binary, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

        if (contours.empty()) {
            return image; // Return original on failure
        }

        // Find the largest contour
        double max_area = 0;
        vector<Point> largest_contour;
        for (const auto& contour : contours) {
            double area = contourArea(contour);
            if (area > max_area) {
                max_area = area;
                largest_contour = contour;
            }
        }

        Rect bounding_box = boundingRect(largest_contour);
        Mat cropped = binary(bounding_box);
        return cropped;
    } else {
        cout << "Detected dark background, skipping centering to preserve data." << endl;
        return image;
    }
}
    
Mat ImagePreprocessor::resizeToMNIST(const Mat& image) {
    Mat resized;
    
    resize(image, resized, Size(28, 28), 0, 0, INTER_AREA);
    
    return resized;
}

Mat ImagePreprocessor::invertColors(const Mat& image) {
    Mat inverted;
    
    bitwise_not(image, inverted);
    
    return inverted;
}

vector<float> ImagePreprocessor::matToFloat(const Mat& image) {
    vector<float> result(784);
    
    if (image.rows != 28 || image.cols != 28 || image.channels() != 1) {
        throw runtime_error("Image must be 28x28 grayscale for MNIST");
    }
    
    for (int y = 0; y < 28; y++) {
        for (int x = 0; x < 28; x++) {
            unsigned char pixel = image.at<unsigned char>(y, x);
            result[y * 28 + x] = static_cast<float>(pixel) / 255.0f;
        }
    }
    
    return result;
}

void ImagePreprocessor::saveDebugImage(const Mat& image, const string& output_path) {
    bool success = imwrite(output_path, image);
    if (!success) {
        cerr << "Warning: Could not save debug image to " << output_path << endl;
    }
}

bool ImagePreprocessor::isImageFile(const string& filename) {
    string extension;
    size_t pos = filename.find_last_of('.');
    if (pos != string::npos) {
        extension = filename.substr(pos);
        transform(extension.begin(), extension.end(), extension.begin(), ::tolower);
    }
    
    return (extension == ".png" || extension == ".jpg" || extension == ".jpeg" ||
            extension == ".bmp" || extension == ".tiff" || extension == ".tif");
}