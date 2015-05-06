#pragma once
#include "TransformImage.hpp"

class HaemoragingImage : public TransformImage
{
public:
    HaemoragingImage(Mat& image) : TransformImage(image, Channels::RED) {}
};