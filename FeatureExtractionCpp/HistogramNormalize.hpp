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
    Mat _refMask;
    vector<Mat> _refHist;

    vector<Channels> _channels;
    bool _hasCalcedHist = false;
    bool _hasCalcedFreqHist = false;

    void CalcRefHistogram()
    {
        if (_hasCalcedHist)
        {
            return;
        }

        for (Channels ch : _channels)
        {
            CalcHist(_refImage, _refHist[(int)ch], ch, _refMask);
        }
        _hasCalcedHist = true;
    }

    void CalcHist(TransformImage& ti, Mat& hist, Channels channel, Mat& mask = Mat())
    {
        int normConst = 0xFF;
        Mat buf(1, normConst, CV_32SC1);
        ti.GetOneChannelImages(channel);
        

        //REVIEW: OpenCV V range is 0 - 255, so simple call to GetHist
        // which is not parameterized should be fine
        ti.GetHist(buf, mask);

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

        float nonZero = countNonZero(mask);
        float normalization = (float)normConst / nonZero;
        int sum = 0;

        for (i = 0; i < buf.rows; i++)
        {
            sum += bufV[i];
            cumsumV[i] = saturate_cast<uchar>(sum * normalization);
        }

        Mat res(cumsumV);
        transpose(res, hist);
    }

    // map inpHist to refHist (in img and ref):
    // |histDest[i] - histRef[j]| = min(k) |histDest[i] - histRef[k]|
    void CreateHistMap(Mat& ref, Mat& img, Mat& dst)
    {
        // copying into vectors makes things dramatically faster
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
                    map[i] = (uchar)j;
                }
            }
        }
        // This creates a 255x1 Mat
        Mat res(map);

        // We want the mapping to be a 1x255 Mat
        transpose(res, dst);
    }

    // Algorithm described here:
    // http://fourier.eng.hmc.edu/e161/lectures/contrast_transform/node3.html
    // if freq is true, ignore channel
    void HistogramSpecification(Mat& image, gpu::GpuMat& dest, Channels channel, Mat& mask)
    {
        //1. Get the reference image histogram (cumulative, normalized)
        CalcRefHistogram();

        TransformImage ti(image);
        Mat hist;

        //2. Get the input image histogram
        CalcHist(ti, hist, channel, mask);
        Mat mapping;

        //3. Create the mapping
        CreateHistMap(_refHist[(int)channel], hist, mapping);

        //4 actually map the pixels
        Mat init = Mat::zeros(image.rows, image.cols, CV_8UC1);
        dest.upload(init);
        gpu::GpuMat gim;
        gim = ti.getChannelImage(channel);

        gpu::LUT(gim, mapping, dest);
    }

public:
    HistogramNormalize(Mat refImage, Mat refMask, vector<Channels> channels) : _refImage(refImage), _refHist(refHistCh, refHistCh + 3), _channels(3), _refMask(refMask)
    { 
        std::copy(channels.begin(), channels.end(), _channels.begin());
    }

    // calculate the histogram for the entire image
    // with "flattened" values of the 3 channels
 

    void HistogramSpecification(Mat& image, Mat& dest, Mat& mask = Mat())
    {
        gpu::GpuMat chImg[3] = { gpu::GpuMat(), gpu::GpuMat(), gpu::GpuMat() };
        for (Channels channel : _channels)
        {
            int i = (int)channel;
            HistogramSpecification(image, chImg[i], channel, mask);
        }

        gpu::GpuMat gDest;
        gpu::merge(chImg, 3, gDest);
        gDest.download(dest);
    }
};