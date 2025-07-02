#ifndef VISUALIZE_H
#define VISUALIZE_H

#include <vector>

void showImage(const vector<unsigned char>& image, int label, int index);

void showMultipleImages(const vector<vector<unsigned char>>& images, const vector<unsigned char>& lables, int count = 5);

#endif