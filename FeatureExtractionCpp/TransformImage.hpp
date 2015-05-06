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
    gpu::GpuMat& GetOneChannelImage(Channels channel)
    {
        if ((int) channel <= 2)
        {
            gpu::GpuMat g_split[3];
            gpu::split(g_image, g_split);
            g_split[(int) channel].copyTo(g_oneChannel);
        }
        else
        {
            gpu::cvtColor(g_image, g_oneChannel, COLOR_BGR2GRAY);
        }

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
        GetOneChannelImage(_channel);
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
        gpu::GpuMat g_enhanced_show;

        if (_channel != Channels::GRAY)
        {
            int channel = (int)_channel;

            vector<gpu::GpuMat> g_zeros(3);
            for (int i = 0; i < 3; i++)
            {   
                if (i == channel)
                {
                    g_enhanced.copyTo(g_zeros[i]);
                }
                else
                {
                    g_zeros[0].upload(Mat::zeros(_image.rows, _image.cols, CV_8UC1));
                }
                
            }
            gpu::merge(g_zeros, g_enhanced_show);
        }
        else
        {
            g_enhanced_show = g_enhanced;
        }

        g_enhanced_show.download(enhanced);
        namedWindow("Enhanced", WINDOW_NORMAL);
        imshow("Enhanced", enhanced);
    }

    gpu::GpuMat& AdaptiveThreshold(int blockSize, double param)
    {
        Mat image, out;
        g_enhanced.download(image);
        adaptiveThreshold(image, out, 127, ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY, blockSize, param);
        g_enhanced.upload(out);

        return g_enhanced;

    }

    gpu::GpuMat& OtsuThreshold()
    {
        //gpu::threshold(g_enhanced, g_enhanced, 0, 255, CV_THRESH_BINARY | CV_THRESH_OTSU);
        Mat image, out;
        g_enhanced.download(image);
        cv::threshold(image, out, 0, 255, CV_THRESH_BINARY | CV_THRESH_OTSU);
        g_enhanced.upload(out);

        return g_enhanced;
    }
   
};
