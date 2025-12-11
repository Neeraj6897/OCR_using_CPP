#include <opencv2/opencv.hpp>
#include <iostream>
#include "dataset_loader.h"
#include "visualize.h"

using namespace std;
using namespace cv;

void showImage(const vector<unsigned char>& image, int label, int index) {
    //Create a 28x28 matrix and fill with image data
    Mat img(28, 28, CV_8UC1); //8-bit grayscale
    for (int i=0; i<28*28; i++) {
        img.at<uchar>(i/28, i%28) = image[i];
    }

    //Resize for better viewing
    resize(img, img, Size(280, 280), 0, 0, INTER_NEAREST);

    //Display the image
    string windowName = "Digit " + to_string(label) + " [#" + to_string(index) + "]";
    imshow(windowName, img);
    waitKey(1000); //Wait for key press
    destroyWindow(windowName);
}

void showMultipleImages(const vector<vector<unsigned char>>& images, const vector<unsigned char>& labels, int count) {
    for (int i=0; i<count; i++){
        showImage(images[i], labels[i], i);
    }
}

int main(){
    string imageFilePath = "/mnt/e/MachineLearningAndGenAI/OCR_Project_in_CPP/dataset/t10k-images.idx3-ubyte";
    string labelFilePath = "/mnt/e/MachineLearningAndGenAI/OCR_Project_in_CPP/dataset/t10k-labels.idx1-ubyte";

    auto images = loadImages(imageFilePath);
    auto labels = loadLabels(labelFilePath);

    cout<<"Label: "<<(int)labels[0]<<endl;

   showMultipleImages(images, labels, 5);

    return 0;
}