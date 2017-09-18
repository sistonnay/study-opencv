/*
 * BaseMethods.h
 *
 *  Created on: 2017年9月17日
 *      Author: ForeverApp
 *  h文件中直接定义函数，则该文件只能被引用一次，否则报重复定义错误
 */

#ifndef JNI_BASEMETHODS_H_
#define JNI_BASEMETHODS_H_

#define TAG_BAGE "BaseMethods"

#include "Slog.h"
#include <vector>
#include <iostream>
#include <sys/time.h>
#include <opencv2/opencv.hpp>
#include <opencv2/xfeatures2d.hpp>

using namespace std;
using namespace cv;
using namespace cv::xfeatures2d;

struct SurfParams {
    Mat * src = NULL;
    Mat * dst = NULL;
    vector<KeyPoint> * keypoints = NULL;
    Ptr<Feature2D> * surf = NULL;
    //引用在定义的时候必须被初始化，所以此处只能用指针
    SurfParams(Mat* src, Mat* dst, vector<KeyPoint>* keypoints,
            Ptr<Feature2D> * surf) {
        this->src = src;
        this->dst = dst;
        this->keypoints = keypoints;
        this->surf = surf;
    }
};

char * calTimeSpend(timeval start, timeval end, const char* str) {
    int spend = 1000000 * (end.tv_sec - start.tv_sec) + end.tv_usec
            - start.tv_usec;
    char* deta = new char[32];
    sprintf(deta, "Total spend of %s: %d ms.", str, spend / 1000);
    return deta;
}

void * SURFThread(void * param) {
    // time_t t =time(NULL);
    // time(NULL) 这句返回的只是一个时间cuo, 精度秒
    struct timeval start, end;
    gettimeofday(&start, NULL); // Linux 下

    struct SurfParams * temp;
    temp = (SurfParams *) param;
    (*temp->surf)->detectAndCompute(*temp->src, noArray(), *temp->keypoints,
            *temp->dst);

    gettimeofday(&end, NULL);
    char * spend = calTimeSpend(start, end, "SURF");
    LOG(TAG_BAGE, spend);
    delete (spend);

    pthread_detach(pthread_self());
    return NULL;
}

class BaseMethods {
public:
    static double zoom_default;

public:
    BaseMethods();
    virtual ~BaseMethods();

    static int showImage(const char* img);
    static void resizeImage(Mat& src, Mat& dest, double size, int type);
    static int compareImages(const char* img1, const char* img2);
    static void matchFeatures(Mat& query, Mat& train, vector<DMatch>& matches);
    static void matchFeatures(Mat& query, Mat& train, vector<DMatch>& matches,
            DescriptorMatcher& matcher);
    static bool refineMatchesWithHomography(const vector<KeyPoint>& queryPoints,
            const vector<KeyPoint>& trainPoints, float reprojectionThreshold,
            vector<DMatch>& matches, Mat& homography);

};

double BaseMethods::zoom_default = 0.25;

int BaseMethods::showImage(const char* img) {
    Mat src = imread(img, IMREAD_COLOR), dst;
    if (!src.data) {
        LOG(TAG_BAGE, "No data!--Exiting the program!");
        return -1;
    }
    resizeImage(src, dst, zoom_default, INTER_LINEAR);
    namedWindow("Show Image", CV_WINDOW_AUTOSIZE);
    imshow("Show Image", dst);
    waitKey(0);
    return 1;
}

void BaseMethods::resizeImage(Mat& src, Mat& dest, double size, int type) {
    if (src.data == NULL) {
        LOG(TAG, "Input Mat is null!");
        return;
    }
    //Size re_size = src.size() / size;
    int width = src.size().width * size;
    int heigth = src.size().height * size;
    resize(src, dest, Size(width, heigth), 0, 0, type);
}

int BaseMethods::compareImages(const char* img1, const char* img2) {

    Mat rsrc_1, rsrc_2;
    Mat src_1 = imread(img1), src_2 = imread(img2);

    resizeImage(src_1, rsrc_1, 0.5, INTER_LINEAR);
    resizeImage(src_2, rsrc_2, 0.5, INTER_LINEAR);

    struct timeval start, end;
    gettimeofday(&start, NULL); // Linux 下

    Mat dest_1, dest_2;
    vector<KeyPoint> keypoints_1, keypoints_2;
    Ptr<Feature2D> surf = SURF::create();

    SurfParams surfParam1(&rsrc_1, &dest_1, &keypoints_1, &surf);
    SurfParams surfParam2(&rsrc_2, &dest_2, &keypoints_2, &surf);

    pthread_t th1, th2;
    pthread_create(&th1, NULL, SURFThread, &surfParam1);
    pthread_create(&th2, NULL, SURFThread, &surfParam2);
    pthread_join(th1, NULL);
    pthread_join(th2, NULL);

    // surf->detectAndCompute(rsrc_1, noArray(), keypoints_1, dest_1);
    // surf->detectAndCompute(rsrc_2, noArray(), keypoints_2, dest_2);

    vector<DMatch> matches;
    BFMatcher matcher(NORM_L2);
    // FlannBasedMatcher matcher(NORM_L1);
    matchFeatures(dest_1, dest_2, matches, matcher);
    // Homography Match
    Mat homography;
    refineMatchesWithHomography(keypoints_1, keypoints_2, 3, matches,
            homography);
    cout<< homography <<endl;

    gettimeofday(&end, NULL);
    char * spend = calTimeSpend(start, end, "Match");
    LOG(TAG_BAGE, spend);
    delete (spend);

    Mat match_keypoints;
    drawMatches(rsrc_1, keypoints_1, rsrc_2, keypoints_2, matches,
            match_keypoints, Scalar::all(-1), Scalar::all(-1), vector<char>(),
            DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);

    imshow("Matching Result", match_keypoints);
    //imwrite("match_result.png", match_keypoints);
    waitKey(0);

    return 1;
}

// 与ORB结合使用，效果较好
void BaseMethods::matchFeatures(Mat& query, Mat& train,
        vector<DMatch>& matches) {
    flann::Index flannIndex(query, flann::LshIndexParams(12, 20, 2),
            cvflann::FLANN_DIST_HAMMING);
    Mat matchindex(train.rows, 2, CV_32SC1);
    Mat matchdistance(train.rows, 2, CV_32FC1);
    flannIndex.knnSearch(train, matchindex, matchdistance, 2,
            flann::SearchParams());
    //根据劳氏算法
    for (int i = 0; i < matchdistance.rows; i++) {
        if (matchdistance.at<float>(i, 0)
                < 0.6 * matchdistance.at<float>(i, 1)) {
            DMatch dmatches(matchindex.at<int>(i, 0), i,
                    matchdistance.at<float>(i, 0));
            matches.push_back(dmatches);
        }
    }
}

// 暴力匹配法，此种方法结合sift、surf用的比较多
void BaseMethods::matchFeatures(Mat& query, Mat& train, vector<DMatch>& matches,
        DescriptorMatcher& matcher) {
    vector<vector<DMatch>> temp;
    matcher.knnMatch(query, train, temp, 2);
    //获取满足Ratio Test的最小匹配的距离
    float min_distance = FLT_MAX;
    for (unsigned r = 0; r < temp.size(); ++r) {
        //Ratio Test
        if (temp[r][0].distance > 0.6 * temp[r][1].distance)
            continue;
        float distance = temp[r][0].distance;
        if (distance < min_distance)
            min_distance = distance;
    }
    matches.clear();
    for (size_t r = 0; r < temp.size(); ++r) {
        //排除不满足Ratio Test的点和匹配距离过大的点
        if (temp[r][0].distance > 0.6 * temp[r][1].distance
                || temp[r][0].distance > 5 * max(min_distance, 10.0f))
            continue;
        //保存匹配点
        matches.push_back(temp[r][0]);
    }
}

bool BaseMethods::refineMatchesWithHomography(
        const vector<KeyPoint>& queryPoints,
        const vector<KeyPoint>& trainPoints, float reprojectionThreshold,
        vector<DMatch>& matches, Mat& homography) {
    const int minMatchesNum = 8;
    if (matches.size() < minMatchesNum)
        return false;
    vector<Point2f> srcPoints(matches.size());
    vector<Point2f> dstPoints(matches.size());
    for (size_t i = 0; i < matches.size(); i++) {
        srcPoints[i] = trainPoints[matches[i].trainIdx].pt;
        dstPoints[i] = queryPoints[matches[i].queryIdx].pt;
    }
    // Find homography matrix and get inliers mask
    vector<unsigned char> inliersMask(srcPoints.size());
    homography = findHomography(srcPoints, dstPoints, CV_FM_RANSAC,
            reprojectionThreshold, inliersMask);
    vector<DMatch> inliers;
    for (size_t i = 0; i < inliersMask.size(); i++) {
        if (inliersMask[i])
            inliers.push_back(matches[i]);
    }
    matches.swap(inliers);
    return matches.size() > minMatchesNum;
}

#endif /* JNI_BASEMETHODS_H_ */
