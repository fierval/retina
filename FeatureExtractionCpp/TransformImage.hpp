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
    Mat _enhanced;

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

        // in case we are not doing any further enhancements...
        g_oneChannel.copyTo(g_enhanced);

        return g_enhanced;
    }

    // apply Gaussian Pyramid scale times
    gpu::GpuMat& PyramidDown(int scale = 1)
    {
        gpu::GpuMat buffer;
        g_image.copyTo(buffer);

        for (int i = scale; i > 0; i--, g_enhanced.copyTo(buffer))
        {
            gpu::pyrDown(buffer, g_enhanced);
        }

        return g_enhanced;
    }

    // preprocessing may be different
    gpu::GpuMat& GaussBlur()
    {
        gpu::GaussianBlur(g_oneChannel, g_enhanced, Size(5, 5), 30.);
        return g_enhanced;
    }

    TransformImage(Mat image, Channels selectChannel) : _image(image), _channel(selectChannel)
    {
        g_image.upload(image);
        GetOneChannelImage(_channel);
    }

    TransformImage() {}

    Mat& GetCannyEdges(int thresh, Mat& edges)
    {
        gpu::GpuMat g_edges;
        gpu::Canny(g_enhanced, g_edges, thresh, thresh * 2);

        g_edges.download(edges);
        return edges;
    }

    vector<vector<Point>>& FindBlobContours(Mat& edges)
    {
        findContours(edges, _contours, _hierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE, Point(0, 0));
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

    // slap contours onto an image
    static void DrawContours(vector<vector<Point>>& contours, vector<Vec4i>& hierarchy, Mat& img, int thickness = CV_FILLED)
    {
        int idx = 0;
        if (hierarchy.size() == 0)
        {
            for (idx = 0; idx < contours.size(); idx++)
            {
                Scalar color = Scalar(255, 255, 255);
                drawContours(img, contours, idx, color, thickness, 8, noArray());
            }

        }
        else
        {
            for (; idx >= 0; idx = hierarchy[idx][0])
            {
                Scalar color = Scalar(255, 255, 255);
                drawContours(img, contours, idx, color, thickness, 8, hierarchy);
            }
        }
    }

    // Accessors
    void setImage(Mat& image) { _image = image; g_image.upload(image); }
    void setChannel(Channels channel) { _channel = channel; }
    Mat& getEnhanced() { g_enhanced.download(_enhanced); return _enhanced; }
    vector<Vec4i>& getHierarchy() { return _hierarchy; }
    vector<vector<Point>>& getContours() { return _contours; }

    void setContours(vector<vector<Point>>& contours) 
    { 
        _contours.clear(); 
        _hierarchy.clear();
        _contours.resize(contours.size()); 
        copy(contours.begin(), contours.end(), _contours.begin()); 
    }

};
