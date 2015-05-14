#pragma once
#include "TransformImage.hpp"

class HaemoragingImage : public TransformImage
{
public:
 
    HaemoragingImage(Mat& image) : TransformImage(image) {}
    HaemoragingImage() : TransformImage() {}

    vector<vector<Point>>& FlattenContours()
    {
        Mat black = Mat::zeros(_image.rows, _image.cols, CV_8UC1);
        DrawContours(_contours, _hierarchy, black, 2);
        _contours.clear();
        _hierarchy.clear();

        return FindBlobContours(black);
    }
};