// FeatureExtractionCpp.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include "FindHaemorages.h"

const const char* keys =
{
    "{1 | | | must specify image name}"
};

Mat src, src_gray;
RNG rng(12345);
string sourceWindow("Source");

void thresh_callback(int cookie, void * param)
{
    vector<vector<Point> > contours;
    vector<Vec4i> hierarchy;

    int thresh = *((int *)param);
    ParamBag params;
    params.cannyThresh = thresh;

    FindHaemorages(src_gray, contours, params);

    /// Draw contours
    for (size_t i = 0; i< contours.size(); i++)
    {
        Scalar color = Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255));
        drawContours(src, contours, (int)i, color, 2, 8, hierarchy, 0, Point());
    }

    /// Show in a window
    imshow(sourceWindow, src);

}

int main(int argc, char** argv)
{
    CommandLineParser parser(argc, argv, keys);
    string file_name = parser.get<string>("1");

    src = imread(file_name, IMREAD_COLOR);
    cvtColor(src, src_gray, COLOR_BGR2GRAY);

    vector<vector<Point>> contours;
    ParamBag params;
    params.cannyThresh = 100;

    namedWindow(sourceWindow, WINDOW_NORMAL);
    createTrackbar("Track", sourceWindow, &(params.cannyThresh), params.cannyThresh * 2, thresh_callback);
    thresh_callback(0, &(params.cannyThresh));
}

