#pragma once
#include "TransformImage.hpp"

// histogram specification & equalization for color images
class HistogramNormalize
{

private:
    TransformImage _refImage;
    gpu::GpuMat g_hist;
    Mat _refHist;
    const Channels channels[3] = { Channels::H, Channels::S, Channels::V };
    bool _hasCalcedHist = false;

    void CalcRefHistogram()
    {
        if (_hasCalcedHist)
        {
            return;
        }

        CalcIntensityHist(_refImage, _refHist);
        _hasCalcedHist = true;
    }

public:
    HistogramNormalize(Mat refImage) : _refImage(refImage)  
    { 
    }

    void CalcIntensityHist(TransformImage& ti, Mat& hist)
    {
        ti.MakeHsv();
        ti.GetOneChannelImages(Channels::V);
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
    void HistogramSpecification(Mat& image)
    {
        //1. Get the reference image histogram (cumulative, normalized)
        CalcRefHistogram();

        TransformImage ti(image);
        Mat hist;

        //2. Get the input image histogram
        CalcIntensityHist(ti, hist);
        Mat mapping;
        
        //3. Create the mapping
        CreateHistMap(_refHist, hist, mapping);

        //TODO: 4 actually map the pixels
    }

};