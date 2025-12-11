#ifndef DATASET_LOADER_H
#define DATASET_LOADER_H

#include <vector>
#include <string>
using namespace std;

vector<vector<unsigned char>> loadImages(const string& filename);
vector<unsigned char> loadLabels(const string& filename);
vector <float> flatteningImages(vector<vector<unsigned char>>& images);

#endif