#pragma once
#include "TransformImage.hpp"

class HaemoragingImage : public TransformImage
{
public:
    virtual gpu::GpuMat& PreprocessImage()
    {
       return TransformImage::PreprocessImage();
    }

    HaemoragingImage(Mat& image) : TransformImage(image, Channels::GRAY) {}
    HaemoragingImage() : TransformImage() {}
};