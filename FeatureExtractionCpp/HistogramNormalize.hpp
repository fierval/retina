#include "TransformImage.hpp"
#pragma once

#pragma warning(disable: 4244)

Mat refHistCh[3] = { Mat(), Mat(), Mat() };

// histogram specification & equalization for color images
class HistogramNormalize
{

private:
    TransformImage _refImage;
    gpu::GpuMat g_hist;
    vector<Mat> _refHist;

    vector<Channels> _channels;
    bool _hasCalcedHist = false;
    bool _hasCalcedFreqHist = false;

    void CalcRefHistogram(Channels channel)
    {
        if (_hasCalcedHist)
        {
            return;
        }

        for (Channels ch : _channels)
        {
            CalcHist(_refImage, _refHist[(int)ch], channel);
        }
        _hasCalcedHist = true;
    }

    void ImageMap(Mat& image, Mat& dest, Mat& mapping)
    {
        vector<uchar> mappingV;
        mapping.copyTo(mappingV);

        for (int i = 0; i < image.rows; i++)
        {
            for (int j = 0; j < image.cols; j++)
            {
                dest.at<uchar>(i, j) = mappingV[image.at<uchar>(i, j)];
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

        Mat cumsum(Mat::zeros(buf.size(), CV_8UC1));
        vector<uchar> cumsumV;
        vector<float> bufV;

        cumsum.copyTo(cumsumV);
        buf.copyTo(bufV);

        int i = 0;
        while (!bufV[i]) ++i;

        int total = (int)image.total();
        if (bufV[i] == total)
        {
            hist.setTo(i);
            return;
        }

        float normalization = (float)normConst / total;
        int sum = 0;

        for (i; i < buf.cols; i++)
        {
            sum += bufV[i];
            cumsumV[i] = saturate_cast<uchar>(sum * normalization);
        }

        Mat res(cumsumV);
        transpose(res, hist);
    }

    // map inpHist to refHist (in img and ref):
    // |img[i] - ref[j]| = min(k) |img[i] - ref[k]|
    void CreateHistMap(Mat& ref, Mat& img, Mat& dst)
    {
        dst = Mat::zeros(ref.size(), CV_8UC1);
        vector<uchar> histSource;
        vector<uchar> histRef;
        vector<uchar> map(ref.cols, 0);

        ref.copyTo(histRef);
        img.copyTo(histSource);

        for (int i = 0; i < img.cols; i++)
        {
            uchar curMin = std::abs(histSource[i] - histRef[0]);
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
    HistogramNormalize(Mat refImage, vector<Channels> channels) : _refImage(refImage), _refHist(refHistCh, refHistCh + 3), _channels(3)
    { 
        std::copy(channels.begin(), channels.end(), _channels.begin());
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
        CreateHistMap(_refHist[(int)channel], hist, mapping);

        //4 actually map the pixels
        dest = Mat::zeros(image.rows, image.cols, CV_8UC1);
        Mat im;
        ti.getChannelImage(channel, im);

        ImageMap(im, dest, mapping);

    }
};