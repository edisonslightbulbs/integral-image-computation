#include <opencv2/opencv.hpp>

cv::Mat computeNaive(cv::Mat& image)
{
    cv::Mat iimage = cv::Mat::zeros(image.rows, image.cols, CV_64FC1);
    for (int x = 0; x < image.rows; x++) {
        for (int y = 0; y < image.cols; y++) {
            for (int ix = 0; ix <= x; ix++) {
                for (int iy = 0; iy <= y; iy++) {
                    iimage.at<double>(x, y) += image.at<double>(ix, iy);
                }
            }
        }
    }
    return iimage;
}

cv::Mat computeUsingPrev(cv::Mat& image)
{
    cv::Mat iimage = cv::Mat::zeros(image.rows, image.cols, CV_64FC1);

    int stride = iimage.cols;
    auto* image_ptr = image.ptr<double>(0);

    int istride = image.cols;
    auto* iimage_ptr = iimage.ptr<double>(0);

    for (int x = 0; x < image.rows; x++) {
        int nh = x - 1;
        double yxSum = 0;
        for (int y = 0; y < image.cols; y++) {
            if (nh >= 0) {
                iimage_ptr[y] += iimage_ptr[y - stride];
            }
            yxSum += image_ptr[y];
            iimage_ptr[y] += yxSum;
        }
        iimage_ptr += stride;
        image_ptr += istride;
    }
    return iimage;
}

cv::Mat computeUsingPadding(cv::Mat& image)
{
    cv::Mat iimage = cv::Mat::zeros(image.rows + 1, image.cols + 1, CV_64FC1);

    int istride = iimage.cols;
    auto* iimage_ptr = iimage.ptr<double>(1) + 1; //Start from (1,1)

    int stride = image.cols;
    auto* image_ptr = image.ptr<double>(0);

    for (int x = 0; x < image.rows; x++) {
        int ySum = 0;
        for (int y = 0; y < image.cols; y++) {
            ySum += (int)image_ptr[y];
            iimage_ptr[y] = ySum + iimage_ptr[y - istride];
        }
        image_ptr += stride;
        iimage_ptr += istride;
    }
    iimage = iimage(cv::Rect(1, 1, image.cols, image.rows));
    return iimage;
}

int main()
{
    cv::TickMeter timer;
    cv::Mat sample = cv::Mat::eye(200, 200, CV_64FC1);

    timer.start();
    computeNaive(sample);
    timer.stop();
    std::cout << "computeNaive (200*200): " << timer << std::endl;

    timer.reset();
    timer.start();
    computeUsingPrev(sample);
    timer.stop();
    std::cout << "computeUsingPrev (200*200): " << timer << std::endl;

    timer.reset();
    timer.start();
    computeUsingPadding(sample);
    timer.stop();
    std::cout << "computeUsingPadding (200*200): " << timer << std::endl;

    sample = cv::Mat::eye(5000, 5000, CV_64FC1);
    timer.reset();
    timer.start();
    computeUsingPrev(sample);
    timer.stop();
    std::cout << "computeUsingPrev (5000*5000): " << timer << std::endl;

    timer.reset();
    timer.start();
    computeUsingPadding(sample);
    timer.stop();
    std::cout << "computeUsingPadding (5000*5000): " << timer << std::endl;
}
