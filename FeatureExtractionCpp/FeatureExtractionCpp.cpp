// FeatureExtractionCpp.cpp : Defines the entry point for the console application.
//
#include "stdafx.h"

#include "HaemoragingImage.hpp"
#include "HistogramNormalize.hpp"


const char* keys =
{
    "{1||| must specify reference image name}"
    "{2||| must specify image name}"

};

Mat src, src_gray, reference;
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
    string ref_file_name = parser.get<string>("1");
    string file_name = parser.get<string>("2");

    gpu::printCudaDeviceInfo(0);

    Mat rgb;
    rgb = imread(file_name, IMREAD_COLOR);

    auto hi = HaemoragingImage(rgb);
    hi.PyramidDown();
    Mat src = hi.getEnhanced();
    rgb = imread(ref_file_name, IMREAD_COLOR);

    auto ref_image = HaemoragingImage(rgb);
    ref_image.PyramidDown();
    reference = ref_image.getEnhanced();

    Channels _channels[3] = { Channels::RED, Channels::GREEN, Channels::BLUE};
    vector<Channels> channels(_channels, _channels + 3);

    auto histSpec = HistogramNormalize(reference, channels);

    Mat dest;
    histSpec.HistogramSpecification(src, dest);

    namedWindow(sourceWindow, WINDOW_NORMAL);
    imshow(sourceWindow, dest);

    //params.cannyThresh = 30;

    //createTrackbar("Track", sourceWindow, &(params.cannyThresh), 100, thresh_callback);
    //thresh_callback(0, &(params.cannyThresh));
    //ref_image.DisplayEnhanced(true);
    waitKey(0);
    return(0);
}

