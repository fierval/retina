#include "stdafx.h"

#include "HaemoragingImage.hpp"

vector<vector<Point>>& FindHaemorages(Mat& image, ParamBag& params)
{
    HaemoragingImage haemorage(image);
    haemorage.PreprocessImage();
    haemorage.DisplayEnhanced();

    return haemorage.FindBlobContours(params.cannyThresh);
}