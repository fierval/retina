#include "stdafx.h"

#include "HaemoragingImage.hpp"

vector<vector<Point>>& FindHaemorages(unique_ptr<HaemoragingImage>& haemorage, Mat& image, ParamBag& params)
{
    haemorage->setImage(image);
    haemorage->setChannel(Channels::GRAY);
    haemorage->GetOneChannelImage(Channels::GRAY);

    haemorage->PreprocessImage();
    haemorage->DisplayEnhanced();

    return haemorage->FindBlobContours(params.cannyThresh);
}