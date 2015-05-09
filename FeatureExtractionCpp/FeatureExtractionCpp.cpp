// FeatureExtractionCpp.cpp : Defines the entry point for the console application.
//
#include "stdafx.h"
#include "HaemoragingImage.hpp"

const char* keys =
{
    "{1||| must specify image name}"
};

Mat src, src_gray;
RNG rng(12345);
string sourceWindow("Source");

ParamBag params;
unique_ptr<HaemoragingImage> haemorage(new HaemoragingImage);

vector<vector<Point>> * FindHaemorages(unique_ptr<HaemoragingImage>&, Mat&, ParamBag&);

void thresh_callback(int, void *)
{
    unique_ptr<vector<vector<Point>>> contours(FindHaemorages(haemorage, src, params));

    Mat img;
    src.copyTo(img);
    TransformImage::DrawContours(*contours, vector<Vec4i>(), img);

    /// Draw contours
    /// Show in a window
    imshow(sourceWindow, img);
}

int main(int argc, char** argv)
{
    CommandLineParser parser(argc, argv, keys);
    string file_name = parser.get<string>("1");

    gpu::printCudaDeviceInfo(0);

    src = imread(file_name, IMREAD_COLOR);
    auto hi = HaemoragingImage(src);
    hi.PyramidDown(2);
    src = hi.getEnhanced();

    params.cannyThresh = 23;

    namedWindow(sourceWindow, WINDOW_NORMAL);
    createTrackbar("Track", sourceWindow, &(params.cannyThresh), 100, thresh_callback);
    thresh_callback(0, &(params.cannyThresh));
    waitKey(0);
    return(0);
}

