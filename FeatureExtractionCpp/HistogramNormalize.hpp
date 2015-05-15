#pragma once
#include "stdafx.h"
#include "TransformImage.hpp"

#pragma warning(disable: 4244)
const Channels default_channels[3] = { Channels::H, Channels::S, Channels::V };

// histogram specification & equalization for color images
class HistogramNormalize
{

private:
    TransformImage _refImage;
    gpu::GpuMat g_hist;
    Mat _refHist;

    const vector<Channels> channels;
    bool _hasCalcedHist = false;
    bool _hasCalcedFreqHist = false;

    void CalcRefHistogram(Channels channel)
    {
        if (_hasCalcedHist)
        {
            return;
        }

        CalcHist(_refImage, _refHist, channel);
        _hasCalcedHist = true;
    }

    void ImageMap(Mat& image, Mat& dest, Mat& mapping)
    {

        for (int i = 0; i < image.rows; i++)
        {
            for (int j = 0; j < image.cols; j++)
            {
                dest.at<uchar>(i, j) = mapping.at<uchar>(0, image.at<T>(i, j));
            }
        }

    }

    void CalcHist(TransformImage& ti, Mat& hist, Channels channel)
    {
        int normConst = 0xFF;
        Mat buf(1, normConst, CV_32SC1);
        ti.GetOneChannelImages(channel);
        gpu::GpuMat gbuf;

        //REVIEW: OpenCV V range is 0 - 255, so simple call to GetHist
        // which is not parameterized should be fine
        ti.GetHist(gbuf);
        gbuf.download(buf);

        Mat image;
        ti.getChannelImage(channel, image);

        Mat cumsum(Mat::zeros(buf.size(), CV_32SC1));
        vector<float> cumsumV, bufV;
        cumsum.copyTo(cumsumV);
        buf.copyTo(bufV);

        float normalization = (float)normConst / (image.rows * image.cols);
        cumsumV[0] = bufV[0] * normalization;

        for (int i = 1; i < buf.cols; i++)
        {
            cumsumV[i] = cumsumV[i - 1] + bufV[i] * normalization;
        }
        Mat res(cumsumV);

        res.copyTo(hist);
    }

    // map inpHist to refHist (in img and ref):
    // |img[i] - ref[j]| = min(k) |img[i] - ref[k]|
    void CreateHistMap(Mat& ref, Mat& img, Mat& dst)
    {
        dst = Mat::zeros(ref.size(), CV_8UC1);
        vector<float> histSource;
        vector<float> histRef;
        ref.copyTo(histRef);
        img.copyTo(histSource);

        for (int i = 0; i < img.cols; i++)
        {
            float curMin = std::abs(histSource[0] - histRef[0]);
            for (int j = 1; j < ref.cols; j++)
            {
                int diff = std::abs(histSource[i] - histRef[j]);
                if (diff < curMin)
                {
                    curMin = diff;
                    dst.at<uchar>(0, i) = (uchar)j;
                }
            }
        }
    }

public:
    HistogramNormalize(Mat refImage) : _refImage(refImage), channels(default_channels, default_channels + 3)
    { 
        
    }

    // calculate the histogram for the entire image
    // with "flattened" values of the 3 channels
 
    // Algorithm described here:
    // http://fourier.eng.hmc.edu/e161/lectures/contrast_transform/node3.html
    // if freq is true, ignore channel
    void HistogramSpecification(Mat& image, Mat& dest, Channels channel)
    {
        //1. Get the reference image histogram (cumulative, normalized)
        CalcRefHistogram(channel);

        TransformImage ti(image);
        Mat hist;

        //2. Get the input image histogram
        CalcHist(ti, hist, channel);
        Mat mapping;
        
        //3. Create the mapping
        CreateHistMap(_refHist, hist, mapping);

        //4 actually map the pixels
        dest = Mat::zeros(image.rows, image.cols, CV_8UC1);

        ImageMap(image, dest, mapping);

    }
};