#include <iostream>
#include "normalize.h"

void normalizeImage(vector<float>& flat_images, size_t offset, size_t image_size) {
    for (size_t i = 0; i < image_size; ++i) {
        flat_images[offset + i] /= 255.0f;
    }
}