#include "stdafx.h"
#pragma once

typedef struct
{
    int cannyThresh;
    int blockSize;
} ParamBag;

enum struct Channels
{
    BLUE = 0,
    GREEN = 1,
    RED = 2,
    GRAY = 3,
    H = 0,
    S = 1,
    V = 2,
    ALL = 4,
};


class TransformImage
{
protected:
    Mat _image;
    Mat _enhanced;

    gpu::GpuMat g_image;
    vector<gpu::GpuMat> g_oneChannel;
    gpu::GpuMat g_enhanced;
    gpu::GpuMat g_buf; //bufffer for different operations
    Channels _channel;
    vector<Vec4i> _hierarchy;

    vector<vector<Point>> _contours;
    inline void MakeSafe() { g_enhanced.copyTo(g_buf); }

public:
    // Split into 3 channels (unless it's grey - then need to use another func)
    // stores one of them in the "enhanced"
    gpu::GpuMat& GetOneChannelImages(Channels channelEnhanced)
    {
        assert(channelEnhanced != Channels::GRAY);

        gpu::split(g_image, g_oneChannel);
        g_oneChannel[(int) channelEnhanced].copyTo(g_enhanced);
        _channel = channelEnhanced;

        return g_enhanced;
    }

    gpu::GpuMat& GetGrayChannelImage()
    {
        _channel = Channels::GRAY;
        gpu::cvtColor(g_image, g_enhanced, COLOR_BGR2GRAY);
        return g_enhanced;
    }

    void MakeHsv()
    {
        g_image.copyTo(g_buf);

        gpu::cvtColor(g_buf, g_image, COLOR_BGR2HSV);
    }

    void MakeBGR()
    {
        g_image.copyTo(g_buf);

        gpu::cvtColor(g_buf, g_image, COLOR_HSV2BGR);
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
    gpu::GpuMat& GaussBlur(int size = 7, float sigma = 30.)
    {
        MakeSafe();

        gpu::GaussianBlur(g_buf, g_enhanced, Size(size, size), sigma);
        return g_enhanced;
    }

    Mat& MedianBlur(int size = 7)
    {
        Mat buf;
        g_enhanced.download(buf);
        medianBlur(buf, _enhanced, size);

        g_enhanced.upload(_enhanced);
        return _enhanced;
    }

    // TODO: this is actually bad since it may throw
    TransformImage(Mat image) : _image(image), g_oneChannel(3)
    {
        g_image.upload(image);
    }

    TransformImage() : g_oneChannel(3)
    {}

    Mat& GetCannyEdges(int thresh, int thresh1, Mat& edges)
    {
        gpu::GpuMat g_edges;
        gpu::Canny(g_enhanced, g_edges, thresh, thresh1);

        g_edges.download(edges);
        return edges;
    }

    vector<vector<Point>>& FindBlobContours(Mat& edges)
    {
        findContours(edges, _contours, _hierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);
        return _contours;
    }

    Mat& GetHist(Mat& dest, Mat& mask = Mat())
    {
        if (mask.size() == Size(0, 0))
        {
            // for once use g_buf to avoid memory allocations
            gpu::GpuMat gDest;
            calcHist(g_enhanced, gDest, g_buf);
            gDest.download(dest);
            return dest;
        }

        Mat enhanced[] = { getEnhanced() };
        int histSize[] = { 256 };
        float hranges[] = { 0, 255 };
        const float * ranges[] = { hranges };
        int channels[] = { 0 };

        calcHist(enhanced, 1, channels, mask, dest, 1, histSize, ranges);
        return dest;
    }

    gpu::GpuMat& ApplyClahe()
    {
        Ptr<gpu::CLAHE> clahe = gpu::createCLAHE();

        clahe->setClipLimit(4.);
        clahe->setTilesGridSize(Size(16, 16));
        clahe->apply(g_oneChannel[(int)_channel], g_enhanced);
        
        g_enhanced.copyTo(g_oneChannel[(int)_channel]);
        
        gpu::merge(g_oneChannel, g_enhanced);
        return g_enhanced;
    }

    // display enhanced image. Do we need to merge all 3 channels to do it?
    // if not, only the relevant channel will be isolated
    void DisplayEnhanced(bool merge3 = false)
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
                    if (merge3)
                    {
                        g_oneChannel[i].copyTo(g_zeros[i]);
                    }
                    else
                    {
                        g_zeros[i].upload(Mat::zeros(_image.rows, _image.cols, CV_8UC1));
                    }
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
    static void DrawContours(vector<vector<Point>>& contours, vector<Vec4i>& hierarchy, Mat& img, int thickness = CV_FILLED, Scalar& color = Scalar(255, 255, 255))
    {
        int idx = 0;
        if (hierarchy.size() == 0)
        {
            for (idx = 0; idx < contours.size(); idx++)
            {
                drawContours(img, contours, idx, color, thickness, 8, noArray());
            }

        }
        else
        {
            for (; idx >= 0; idx = hierarchy[idx][0])
            {
                drawContours(img, contours, idx, color, thickness, 8, hierarchy);
            }
        }
    }

    // Accessors
    void setImage(Mat& image) {image.copyTo(_image); g_image.upload(image); }
    Mat& getImage() { return _image; }
    void setChannel(Channels channel) { _channel = channel; }
    Mat& getEnhanced() { g_enhanced.download(_enhanced); return _enhanced; }
    void getChannelImage(Channels channel, Mat& dst) { g_oneChannel[(int)channel].download(dst); }
    gpu::GpuMat& getChannelImage(Channels channel) { return g_oneChannel[(int)channel]; }
    void setChannelImage(Channels channel, Mat& src) { g_oneChannel[(int)channel].upload(src); }

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
