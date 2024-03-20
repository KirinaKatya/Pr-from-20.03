#include <iostream>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

using namespace cv;
using namespace std;

const int MAX_VALUE = 255;

void grayscale(const Mat& inputImage, Mat& outputImage) {
    int rows = inputImage.rows;
    int cols = inputImage.cols;

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            Vec3b pixel = inputImage.at<Vec3b>(i, j);
            int b = pixel[0];
            int g = pixel[1];
            int r = pixel[2];
            uchar grayValue = (r + g + b) / 3;
            outputImage.at<Vec3b>(i, j) = Vec3b(grayValue, grayValue, grayValue);
        }
    }
}

void sepia(const Mat& inputImage, Mat& outputImage) {
    int rows = inputImage.rows;
    int cols = inputImage.cols;

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            Vec3b pixel = inputImage.at<Vec3b>(i, j);
            int b = pixel[0];
            int g = pixel[1];
            int r = pixel[2];
            int sepiaR = (int)(0.393 * b + 0.769 * g + 0.189 * r);
            int sepiaG = (int)(0.349 * b + 0.686 * g + 0.168 * r);
            int sepiaB = (int)(0.272 * b + 0.534 * g + 0.131 * r);
            outputImage.at<Vec3b>(i, j) = Vec3b(min(sepiaB, MAX_VALUE), min(sepiaG, MAX_VALUE), min(sepiaR, MAX_VALUE));
        }
    }
}

void negative(const Mat& inputImage, Mat& outputImage) {
    int rows = inputImage.rows;
    int cols = inputImage.cols;

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            Vec3b pixel = inputImage.at<Vec3b>(i, j);
            outputImage.at<Vec3b>(i, j) = Vec3b(MAX_VALUE - pixel[0], MAX_VALUE - pixel[1], MAX_VALUE - pixel[2]);
        }
    }
}

void contour(const Mat& inputImage, Mat& outputImage) {
    Mat grayImage;
    cvtColor(inputImage, grayImage, COLOR_BGR2GRAY);
    Mat edges = Mat(inputImage.rows, inputImage.cols, CV_8U);

    for (int i = 1; i < grayImage.rows - 1; i++) {
        for (int j = 1; j < grayImage.cols - 1; j++) {
            float gx = grayImage.at<uchar>(i + 1, j + 1) + 2 * grayImage.at<uchar>(i, j + 1) + grayImage.at<uchar>(i - 1, j + 1) - grayImage.at<uchar>(i + 1, j - 1) - 2 * grayImage.at<uchar>(i, j - 1) - grayImage.at<uchar>(i - 1, j - 1);
            float gy = grayImage.at<uchar>(i + 1, j + 1) + 2 * grayImage.at<uchar>(i + 1, j) + grayImage.at<uchar>(i + 1, j - 1) - grayImage.at<uchar>(i - 1, j - 1) - 2 * grayImage.at<uchar>(i - 1, j) - grayImage.at<uchar>(i - 1, j + 1);
            edges.at<uchar>(i, j) = 255 - sqrt(pow(gx, 2) + pow(gy, 2));
        }
    }

    outputImage = edges.clone();
}


int main() {
    Mat image = imread("C:/Users/Катя/Desktop/Python/source_mat.jpg");

    if (image.empty()) {
        cout << "Not open!" << endl;
        return -1;
    }
    Mat originalImage = image.clone();
    Mat grayscaleImage = originalImage.clone();
    Mat sepiaImage = originalImage.clone();
    Mat negImage = originalImage.clone();
    Mat contourImage = originalImage.clone();

#pragma omp parallel sections num_threads(4)
    {
#pragma omp section
        {
            grayscale(originalImage, grayscaleImage);
        }
#pragma omp section
        {
            sepia(originalImage, sepiaImage);
        }
#pragma omp section
        {
            negative(originalImage, negImage);
        }
#pragma omp section
        {
            contour(originalImage, contourImage);
        }
    }

    namedWindow("original", WINDOW_NORMAL);
    imshow("original", originalImage);

    namedWindow("grayscale", WINDOW_NORMAL);
    imshow("grayscale", grayscaleImage);

    namedWindow("sepia", WINDOW_NORMAL);
    imshow("sepia", sepiaImage);

    namedWindow("negative", WINDOW_NORMAL);
    imshow("negative", negImage);

    namedWindow("contour", WINDOW_NORMAL);
    imshow("contour", contourImage);

    waitKey(0);

    return 0;
}
