#pragma once
#include "TransformImage.hpp"

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

    void CalcRefHistogram(Channels channel)
    {
        if (_hasCalcedHist)
        {
            return;
        }

        CalcHist(_refImage, _refHist, channel);
        _hasCalcedHist = true;
    }

public:
    HistogramNormalize(Mat refImage) : _refImage(refImage), channels(default_channels, default_channels + 3)
    { 
        
    }

    void CalcHist(TransformImage& ti, Mat& hist, Channels channel)
    {
        ti.GetOneChannelImages(channel);
        gpu::GpuMat gbuf;
        Mat buf;

        //REVIEW: OpenCV V range is 0 - 255, so simple call to GetHist
        // which is not parameterized should be fine
        ti.GetHist(gbuf);
        gpu::normalize(gbuf, gbuf, 255., 0., NORM_L1);
        gbuf.download(buf);

        integral(buf, hist);
    }

    // map inpHist to refHist (in img and ref):
    // |img[i] - ref[j]| = min(k) |img[i] - ref[k]|
    void CreateHistMap(Mat& ref, Mat& img, Mat& dst)
    {
        dst = Mat::zeros(ref.size(), CV_8UC1);

        for (int i = 0; i < img.cols; i++)
        {
            long curMin = img.at<long>(0, i) - ref.at<long>(0, 0);
            for (int j = 1; j < ref.cols; j++)
            {
                int diff = std::abs(img.at<long>(0, i) - ref.at<long>(0, j));
                if (diff < curMin)
                {
                    curMin = diff;
                    dst.at<uchar>(0, i) = (uchar)j;
                }
            }
        }
    }

    // Algorithm described here:
    // http://fourier.eng.hmc.edu/e161/lectures/contrast_transform/node3.html
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
        Mat oneChannel;
        ti.GetOneChannelImages(channel).download(oneChannel);

        for (int i = 0; i < image.rows; i++)
        {
            for (int j = 0; j < image.cols; j++)
            {
                dest.at<uchar>(i, j) = hist.at<uchar>(0, image.at<uchar>(i, j));
            }
        }
    }

};