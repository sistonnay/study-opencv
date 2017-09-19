/*
 * BaseMethods.h
 *
 *  Created on: 2017��9��17��
 *      Author: ForeverApp
 *  h�ļ���ֱ�Ӷ��庯��������ļ�ֻ�ܱ�����һ�Σ������ظ��������
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
    //�����ڶ����ʱ����뱻��ʼ�������Դ˴�ֻ����ָ��
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
    // time(NULL) ��䷵�ص�ֻ��һ��ʱ��cuo, ������
    struct timeval start, end;
    gettimeofday(&start, NULL); // Linux ��

    struct SurfParams * temp;
    temp = (SurfParams *) param;
    (*temp->surf)->detectAndCompute(*temp->src, noArray(), *temp->keypoints,
            *temp->dst);

    gettimeofday(&end, NULL);

    LOG(TAG_BAGE, "Total spend of %s: %d ms.", "SURF",
            (int )(1000 * (end.tv_sec - start.tv_sec)
                    + (end.tv_usec - start.tv_usec) / 1000));

    pthread_detach(pthread_self());
    return NULL;
}

class BaseMethods {
public:
    static double dual_camera_f; //��ͬ˫�㣬���࣬��
    static double dual_camera_t; //���ľ��룬��
    static double dual_camera_d; //�ֱ��� ��
    static double zoom_size; //�Ŵ���
    static double density; // ʵ�ʷֱ���

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
    static void calculateDistance(const vector<KeyPoint>& queryPoints,
            const vector<KeyPoint>& trainPoints, const vector<DMatch>& matches,
            vector<double> & distances);
};

double BaseMethods::zoom_size = 0.5; // 1
double BaseMethods::dual_camera_f = 3.75 / 1000; // m
double BaseMethods::dual_camera_t = 21.0 / 1000; // m
double BaseMethods::dual_camera_d = 2 * 1.25 / 1000000; // m, ��ͼ���������
double BaseMethods::density = dual_camera_d;

int BaseMethods::showImage(const char* img) {
    Mat src = imread(img, IMREAD_COLOR), dst;
    if (!src.data) {
        LOG(TAG_BAGE, "%s", "No data!--Exiting the program!");
        return -1;
    }
    resizeImage(src, dst, zoom_size, INTER_LINEAR);
    namedWindow("Show Image", CV_WINDOW_AUTOSIZE);
    imshow("Show Image", dst);
    waitKey(0);
    return 1;
}

void BaseMethods::resizeImage(Mat& src, Mat& dest, double size, int type) {
    if (src.data == NULL) {
        LOG(TAG_BAGE, "%s", "Input Mat is null!");
        return;
    }
    density = dual_camera_d / size;
    LOG(TAG_BAGE, "Current density is %.7f.", density);
    //Size re_size = src.size() / size;
    int width = src.size().width * size;
    int heigth = src.size().height * size;
    resize(src, dest, Size(width, heigth), 0, 0, type);
}

int BaseMethods::compareImages(const char* img1, const char* img2) {

    Mat rsrc_query, rsrc_train;
    Mat src_query = imread(img1), src_train = imread(img2);

    resizeImage(src_query, rsrc_query, zoom_size, INTER_LINEAR);
    resizeImage(src_train, rsrc_train, zoom_size, INTER_LINEAR);

    struct timeval start, end;
    gettimeofday(&start, NULL); // Linux ��

    Mat dest_query, dest_train;
    vector<KeyPoint> keypoints_query, keypoints_train;
    Ptr<Feature2D> surf = SURF::create();

    SurfParams surfParam1(&rsrc_query, &dest_query, &keypoints_query, &surf);
    SurfParams surfParam2(&rsrc_train, &dest_train, &keypoints_train, &surf);

    pthread_t th1, th2;
    pthread_create(&th1, NULL, SURFThread, &surfParam1);
    pthread_create(&th2, NULL, SURFThread, &surfParam2);
    pthread_join(th1, NULL);
    pthread_join(th2, NULL);

    // surf->detectAndCompute(rsrc_query, noArray(), keypoints_query, dest_query);
    // surf->detectAndCompute(rsrc_train, noArray(), keypoints_train, dest_train);

    vector<DMatch> matches;
    BFMatcher matcher(NORM_L2);
    // FlannBasedMatcher matcher(NORM_L1);
    matchFeatures(dest_query, dest_train, matches, matcher);
    // Homography Match
    Mat homography; // ͸�����
    refineMatchesWithHomography(keypoints_query, keypoints_train, 3, matches,
            homography);

    LOG(TAG_BAGE, "%s", "͸�����;");
    cout << homography << endl; // ͸�Ӿ���һ���㵽��һ����ı任��������任

//    -- Get the corners from the img1
//    vector<Point2f> corners_query(4);
//    corners_query[0] = Point2f(0, 0);
//    corners_query[1] = Point2f(rsrc_query.cols, 0);
//    corners_query[2] = Point2f(rsrc_query.cols, rsrc_query.rows);
//    corners_query[3] = Point2f(0, rsrc_query.rows);
//
//    cout << endl << corners_query << endl;
//
//    vector<Point2f> corners_train(4);
//    vector<Point2f> corners_query(4);
//    perspectiveTransform(corners_query, corners_train, homography);

//    cout << endl << corners_train << endl;

    vector<double> distances;
    LOG(TAG_BAGE, "%s", "ƥ���ľ������;");
    calculateDistance(keypoints_query, keypoints_train, matches, distances);

    gettimeofday(&end, NULL);
    LOG(TAG_BAGE, "Total spend of %s: %d ms.", "Compare",
            (int )(1000 * (end.tv_sec - start.tv_sec)
                    + (end.tv_usec - start.tv_usec) / 1000));

    Mat match_keypoints;
    drawMatches(rsrc_query, keypoints_query, rsrc_train, keypoints_train,
            matches, match_keypoints, Scalar::all(-1), Scalar::all(-1),
            vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);

    // cout << keypoints_query[matches[0].queryIdx].pt << endl;
    // cout << keypoints_train[matches[0].trainIdx].pt << endl;
    imshow("Matching Result", match_keypoints);
    // imwrite("match_result.png", match_keypoints);
    waitKey(0);

    return 1;
}

void BaseMethods::calculateDistance(const vector<KeyPoint>& queryPoints,
        const vector<KeyPoint>& trainPoints, const vector<DMatch>& matches,
        vector<double> & distances) {
    double distance;
    for (unsigned int i = 0; i < matches.size(); ++i) {
        distance = dual_camera_f * dual_camera_t / density
                / (queryPoints[matches[i].queryIdx].pt.x
                        - trainPoints[matches[i].trainIdx].pt.x);
        distances.push_back(distance);
        cout << distance << "m";
        if ((i + 1) % 8 != 0) {
            cout << " ";
        } else
            cout << endl;
    }
    cout << endl;
}

// ��ORB���ʹ�ã�Ч���Ϻ�
void BaseMethods::matchFeatures(Mat& query, Mat& train,
        vector<DMatch>& matches) {
    flann::Index flannIndex(query, flann::LshIndexParams(12, 20, 2),
            cvflann::FLANN_DIST_HAMMING);
    Mat matchindex(train.rows, 2, CV_32SC1);
    Mat matchdistance(train.rows, 2, CV_32FC1);
    flannIndex.knnSearch(train, matchindex, matchdistance, 2,
            flann::SearchParams());
//���������㷨
    for (int i = 0; i < matchdistance.rows; i++) {
        if (matchdistance.at<float>(i, 0)
                < 0.6 * matchdistance.at<float>(i, 1)) {
            DMatch dmatches(matchindex.at<int>(i, 0), i,
                    matchdistance.at<float>(i, 0));
            matches.push_back(dmatches);
        }
    }
}

// ����ƥ�䷨�����ַ������sift��surf�õıȽ϶�
void BaseMethods::matchFeatures(Mat& query, Mat& train, vector<DMatch>& matches,
        DescriptorMatcher& matcher) {
    vector<vector<DMatch>> temp;
    matcher.knnMatch(query, train, temp, 2);
//��ȡ����Ratio Test����Сƥ��ľ���
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
        //�ų�������Ratio Test�ĵ��ƥ��������ĵ�
        if (temp[r][0].distance > 0.6 * temp[r][1].distance
                || temp[r][0].distance > 5 * max(min_distance, 10.0f))
            continue;
        //����ƥ���
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
        srcPoints[i] = queryPoints[matches[i].queryIdx].pt; // ����ͼ��ؼ���
        dstPoints[i] = trainPoints[matches[i].trainIdx].pt; // ƥ��ͼ��ؼ���
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
