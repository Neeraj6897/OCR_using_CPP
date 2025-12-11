#include "dataset_loader.h"
#include <iostream>
#include <fstream>
using namespace std;

int readBigEndianInt(ifstream &file) {
    unsigned char b1, b2, b3, b4; 
    file.read((char*)&b1, 1);
    file.read((char*)&b2, 1);
    file.read((char*)&b3, 1);
    file.read((char*)&b4, 1);

    return (b1 << 24) | (b2 << 16) | (b3 << 8) | b4; 
    }
    
//Loading the images converted images from ubyte format
vector<vector<unsigned char>> loadImages(const string& filename) {
    ifstream file(filename, ios::binary); 
    if (!file.is_open()) {
        throw runtime_error("Could not open image file.");
    }

    int magic = readBigEndianInt(file);
    int numImages = readBigEndianInt(file);
    int numRows = readBigEndianInt(file);
    int numCols = readBigEndianInt(file);

    int imageSize = numRows * numCols;

    vector<vector<unsigned char>> images(numImages, vector<unsigned char>(imageSize));

    for(int i=0; i<numImages; i++){
        file.read((char*)images[i].data(), imageSize);
    }

    return images;
}

vector<unsigned char> loadLabels(const string& filename){
    ifstream file(filename, ios::binary);
    if (!file.is_open()) {
        throw runtime_error("Could not open label file.");
    }

    int magic = readBigEndianInt(file);
    int numLabels = readBigEndianInt(file);

    vector<unsigned char> labels(numLabels);
    file.read((char*)labels.data(), numLabels);
    return labels;
}

vector <float> flatteningImages(vector<vector<unsigned char>>& images) {
    vector<float> flatenned_image;
    
    if (!images.empty()) {
        flatenned_image.reserve(images.size() * images[0].size());
    }
    for (const auto& image : images) {
        for (unsigned char pixel : image) {
            flatenned_image.push_back(static_cast<float>(pixel));
        }
    }
    return flatenned_image;

} 