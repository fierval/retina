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

    void CalcRefHistogram()
    {
        if (_hasCalcedHist)
        {
            return;
        }

        for (Channels ch : _channels)
        {
            CalcHist(_refImage, _refHist[(int)ch], ch);
        }
        _hasCalcedHist = true;
    }

    // lookup new values using the mapping matrix
    void ImageMap(Mat& image, Mat& dest, Mat& mapping)
    {

        gpu::GpuMat g_image(image), g_dest(dest);
        gpu::LUT(g_image, mapping, g_dest);
        g_dest.download(dest);
        //vector<uchar> mappingV;
        //mapping.copyTo(mappingV);
        //
        //// using the vector is quicker than accessing a matrix
        //for (int i = 0; i < image.rows; i++)
        //{
        //    int row = i * image.cols;
        //    for (int j = 0; j < image.cols; j++)
        //    {
        //        dest.at<uchar>(i,j) = mappingV[image.at<uchar>(i, j)];
        //    }
        //}
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

        float normalization = (float)normConst / (total - bufV[i]);
        int sum = 0;

        for (i++; i < buf.cols; i++)
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

        Mat res(map);
        transpose(res, dst);
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
        CalcRefHistogram();

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

    void HistogramSpecification(Mat& image, Mat& dest)
    {
        Mat chImg[3] = { Mat(), Mat(), Mat() };
        for (Channels channel : _channels)
        {
            int i = (int)channel;
            HistogramSpecification(image, chImg[i], channel);
        }

        merge(chImg, 3, dest);
    }
};