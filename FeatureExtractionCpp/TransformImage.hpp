#pragma once
#include "stdafx.h"

typedef struct
{
    int cannyThresh;
} ParamBag;

enum struct Channels
{
    BLUE = 0,
    GREEN = 1,
    RED = 2,
    GRAY = 3
};


class TransformImage
{
protected:
    Mat _image;
    gpu::GpuMat g_image;
    gpu::GpuMat g_oneChannel;
    gpu::GpuMat g_enhanced;
    Channels _channel;
    vector<Vec4i> _hierarchy;


    vector<vector<Point>> _contours;

public:
    // in case we want a different kind of image reduction, e.g.: gray scale - override
    virtual gpu::GpuMat& GetOneChannelImage(int channel)
    {
        assert(channel >= 0 && channel <= 2);
        gpu::GpuMat g_split[3];
        gpu::split(g_image, g_split);
        g_split[channel].copyTo(g_oneChannel);
        return g_oneChannel;
    }

    // preprocessing may be different
    virtual gpu::GpuMat& PreprocessImage()
    {
        gpu::GaussianBlur(g_oneChannel, g_enhanced, Size(3, 3), 30.);
        return g_enhanced;
    }

    TransformImage(Mat image, Channels selectChannel) : _image(image), _channel(selectChannel)
    {
        g_image.upload(image);
        GetOneChannelImage((int) _channel);
        PreprocessImage();
    }

    vector<vector<Point>>& FindBlobContours(int thresh)
    {
        gpu::GpuMat g_edges;
        gpu::Canny(g_oneChannel, g_edges, thresh, thresh * 2);
        Mat edges;
        g_edges.download(edges);

        findContours(edges, _contours, _hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0));
        return _contours;
    }

    gpu::GpuMat& ApplyClahe()
    {
        Ptr<gpu::CLAHE> clahe = gpu::createCLAHE();

        clahe->setClipLimit(4.);
        clahe->setTilesGridSize(Size(16, 16));
        clahe->apply(g_oneChannel, g_enhanced);
        return g_enhanced;
    }

    void DisplayEnhanced()
    {
        Mat enhanced;

        vector<gpu::GpuMat> g_zeros(3);
        g_zeros[0].upload(Mat::zeros(_image.rows, _image.cols, CV_8UC1));
        g_zeros[0].copyTo(g_zeros[1]);
        g_enhanced.copyTo(g_zeros[2]);

        gpu::GpuMat g_enhanced_show;

        gpu::merge(g_zeros, g_enhanced_show);

        g_enhanced_show.download(enhanced);
        namedWindow("Enhanced", WINDOW_NORMAL);
        imshow("Enhanced", enhanced);
    }
   
};
