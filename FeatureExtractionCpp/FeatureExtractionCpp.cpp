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

vector<vector<Point>>& FindHaemorages(unique_ptr<HaemoragingImage>&, Mat&, ParamBag&);

void thresh_callback(int, void *)
{
    vector<vector<Point>>& contours = FindHaemorages(haemorage, src, params);
    vector<Vec4i>& hierarchy = haemorage->getHierarchy();

    Mat img;
    src.copyTo(img);
    /// Draw contours
    int idx = 0;
    if (hierarchy.size() == 0)
    {
        Scalar color = Scalar(255, 255, 255);
        drawContours(img, contours, idx, color, 5, 8, noArray());

    }
    else
    {
        for (; idx >= 0; idx = hierarchy[idx][0])
        {
            Scalar color = Scalar(255, 255, 255);
            drawContours(img, contours, idx, color, 5, 8, hierarchy);
        }
    }
    /// Show in a window
    imshow(sourceWindow, img);
}

int main(int argc, char** argv)
{
    CommandLineParser parser(argc, argv, keys);
    string file_name = parser.get<string>("1");

    gpu::printCudaDeviceInfo(0);

    src = imread(file_name, IMREAD_COLOR);
    vector<vector<Point>> contours;

    params.cannyThresh = 100;

    namedWindow(sourceWindow, WINDOW_NORMAL);
    createTrackbar("Track", sourceWindow, &(params.cannyThresh), params.cannyThresh * 2, thresh_callback);
    thresh_callback(0, &(params.cannyThresh));
    waitKey(0);
    return(0);
}

