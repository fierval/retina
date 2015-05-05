#include "stdafx.h"

#include "FindHaemorages.h"

void FindBlobContours(gpu::GpuMat& g_image, vector<vector<Point>>& contours, int thresh)
{
    gpu::GpuMat g_edges;
    gpu::Canny(g_image, g_edges, thresh, thresh * 2);
    Mat edges;
    g_edges.download(edges);
    vector<Vec4i> hierarchy;

    findContours(edges, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0));
}

void EnhanceImage(Mat& grayImage, gpu::GpuMat& g_equalizedImage)
{
    gpu::GpuMat g_grayImage;
    g_grayImage.upload(grayImage);

    Ptr<gpu::CLAHE> clahe = gpu::createCLAHE();

    clahe->setClipLimit(4.);
    clahe->setTilesGridSize(Size(16, 16));
    clahe->apply(g_grayImage, g_equalizedImage);
}

void FindHaemorages(Mat& grayImage, vector<vector<Point>>& contours, ParamBag params)
{
    //1. Enhance the image
    gpu::GpuMat g_enhancedImage;
    EnhanceImage(grayImage, g_enhancedImage);

    //2. Find blobs
    FindBlobContours(g_enhancedImage, contours, params.cannyThresh);
}