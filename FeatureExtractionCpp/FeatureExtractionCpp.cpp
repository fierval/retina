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

    src = imread(file_name, IMREAD_COLOR);
    auto hi = HaemoragingImage(src);
    hi.PyramidDown(2);
    src = hi.getEnhanced();

    auto ref_image = HaemoragingImage(imread(ref_file_name, IMREAD_COLOR));
    ref_image.PyramidDown(2);
    reference = ref_image.getEnhanced();

    auto histSpec = HistogramNormalize(reference);
    vector<Channels> channels(3);
    channels.push_back(Channels::RED);
    channels.push_back(Channels::GREEN);
    channels.push_back(Channels::BLUE);

    for (auto ch : channels)
    {
        Mat img;
        histSpec.HistogramSpecification(src, img, ch);
        ref_image.setChannelImage(ch, img);
    }

    params.cannyThresh = 30;

    namedWindow(sourceWindow, WINDOW_NORMAL);

    //createTrackbar("Track", sourceWindow, &(params.cannyThresh), 100, thresh_callback);
    //thresh_callback(0, &(params.cannyThresh));
    imshow(sourceWindow, reference);
    ref_image.DisplayEnhanced(true);
    waitKey(0);
    return(0);
}

