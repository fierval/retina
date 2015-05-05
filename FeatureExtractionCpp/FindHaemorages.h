#pragma once

typedef struct
{
    int cannyThresh;
} ParamBag;

void FindBlobContours(gpu::GpuMat& g_image, vector<vector<Point>>& contours, int thresh);
void EnhanceImage(Mat& grayImage, gpu::GpuMat& equalizedImage);
void FindHaemorages(Mat& grayImage, vector<vector<Point>>& contours, ParamBag params);


