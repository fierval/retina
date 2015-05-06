// FeatureExtractionCpp.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include "TransformImage.hpp"

const char* keys =
{
    "{1||| must specify image name}"
};

Mat src, src_gray;
RNG rng(12345);
string sourceWindow("Source");

ParamBag params;

vector<vector<Point>>& FindHaemorages(Mat&, ParamBag&);

void thresh_callback(int, void *)
{
    vector<Vec4i> hierarchy;

    vector<vector<Point>>& contours = FindHaemorages(src, params);
    Mat img;
    src.copyTo(img);
    /// Draw contours
    for (size_t i = 0; i< contours.size(); i++)
    {
        Scalar color = Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255));
        drawContours(img, contours, (int)i, color, 2, 8, hierarchy, 0, Point());
    }

    /// Show in a window
    imshow(sourceWindow, img);
}

int main(int argc, char** argv)
{
    CommandLineParser parser(argc, argv, keys);
    string file_name = parser.get<string>("1");

    src = imread(file_name, IMREAD_COLOR);
    vector<vector<Point>> contours;

    params.cannyThresh = 100;

    namedWindow(sourceWindow, WINDOW_NORMAL);
    createTrackbar("Track", sourceWindow, &(params.cannyThresh), params.cannyThresh * 2, thresh_callback);
    thresh_callback(0, &(params.cannyThresh));
    waitKey(0);
    return(0);
}

