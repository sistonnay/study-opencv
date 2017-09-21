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
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/xfeatures2d.hpp>

using namespace std;
using namespace cv;
using namespace cv::xfeatures2d;

struct PointsMBR {
    double left;
    double top;
    double right;
    double bottom;
};

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

    LOG(TAG_BAGE, "Total spend of %s: %d ms.", "SURF",
            (int )(1000 * (end.tv_sec - start.tv_sec)
                    + (end.tv_usec - start.tv_usec) / 1000));

    pthread_detach(pthread_self());
    return NULL;
}

class BaseMethods {
public:
    static double dual_camera_f; //相同双摄，焦距，米
    static double dual_camera_t; //轴心距离，米
    static double dual_camera_d; //分辨率 米
    static double zoom_size; //放大倍数
    static double density; // 实际分辨率
    static double match_threshold; // 匹配点距离阈值， 不能靠的太近

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
            const Mat& query, PointsMBR& mbr, vector<Point3f>& xyzPoints);
    static void doSubDiv2D(const vector<KeyPoint>& queryPoints,
            const vector<DMatch>& matches, const PointsMBR& mbr,
            const Mat& query, Subdiv2D& subdiv);
    static void drowDelaunayPoint(const Point2f point, Subdiv2D& subdiv,
            Mat& img, Scalar color);
    static void drowDelaunayTriangles(const vector<Vec6f>& triangles,
            const PointsMBR& mbr, Mat& img, Scalar color);
    static void drowPointsMBR(const PointsMBR& mbr, Mat& img, Scalar color);
    static bool isPointInMBR(const PointsMBR& mbr, const Point& point);
    static bool isLineInMBR(const PointsMBR& mbr, const Point& point1,
            const Point& point2);
};

double BaseMethods::zoom_size = 1; // 1
double BaseMethods::dual_camera_f = 3.75 / 1000; // m
double BaseMethods::dual_camera_t = 21.0 / 1000; // m
double BaseMethods::dual_camera_d = 2 * 1.25 / 1000000; // m, 成图相机有缩放
double BaseMethods::density = dual_camera_d;
double BaseMethods::match_threshold = 5 * density; // 设置为五个像素大小

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
    gettimeofday(&start, NULL); // Linux 下

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
    Mat homography; // 透射矩阵
    refineMatchesWithHomography(keypoints_query, keypoints_train, 3, matches,
            homography);

    LOG(TAG_BAGE, "%s", "透射矩阵;");
    cout << homography << endl; // 透视矩阵，一个点到另一个点的变换，即仿射变换

    PointsMBR pointsMBR; // -- Get the corners from the img1
    vector<Point3f> points3D;
    LOG(TAG_BAGE, "%s", "匹配点的xyz计算;");
    calculateDistance(keypoints_query, keypoints_train, matches, rsrc_query,
            pointsMBR, points3D);
    LOG(TAG_BAGE, "ImageMBR (%.0f,%.0f,%.0f,%.0f)", pointsMBR.left,
            pointsMBR.top, pointsMBR.right, pointsMBR.bottom);
    //构造OpenCV Rect
    Rect rect(pointsMBR.left - 15, pointsMBR.top - 15,
            pointsMBR.right - pointsMBR.left + 30,
            pointsMBR.bottom - pointsMBR.top + 30);

    Subdiv2D subdiv(rect);
    doSubDiv2D(keypoints_query, matches, pointsMBR, rsrc_query, subdiv);

    gettimeofday(&end, NULL);
    LOG(TAG_BAGE, "Total spend of %s: %d ms.", "Compare",
            (int )(1000 * (end.tv_sec - start.tv_sec)
                    + (end.tv_usec - start.tv_usec) / 1000));

    Mat match_keypoints;
    drawMatches(rsrc_query, keypoints_query, rsrc_train, keypoints_train,
            matches, match_keypoints, Scalar::all(-1), Scalar::all(-1),
            vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);

    imshow("Matching Result", match_keypoints);
    // imwrite("match_result.png", match_keypoints);
    waitKey(0);

    return 1;
}

void BaseMethods::calculateDistance(const vector<KeyPoint>& queryPoints,
        const vector<KeyPoint>& trainPoints, const vector<DMatch>& matches,
        const Mat& query, PointsMBR& mbr, vector<Point3f>& xyzPoints) {
    Point3f xyz;
    Mat temp = query.clone();
    double imgW = temp.cols;
    double imgH = temp.rows;
    double query_x, query_y, train_x, train_y;
    mbr.top = imgH;
    mbr.left = imgW;
    mbr.right = mbr.bottom = 0;

    char text[50];
    int pointsTotal = matches.size(), showPtNum = 30, count = 0;
    int showIntv = max(pointsTotal / showPtNum, 1); // 一共画30个点的坐标

    for (unsigned int i = 0; i < matches.size(); ++i) {
        query_x = queryPoints[matches[i].queryIdx].pt.x;
        query_y = queryPoints[matches[i].queryIdx].pt.y;
        train_x = trainPoints[matches[i].trainIdx].pt.x;
        train_y = trainPoints[matches[i].trainIdx].pt.y;

        xyz.z = dual_camera_f * dual_camera_t / density / (query_x - train_x);
        xyz.y = -density * xyz.z * (query_y + train_y - imgH) / 2
                / dual_camera_f;
        xyz.x = -density * xyz.z * (imgW / 2 - query_x) / dual_camera_f;
        xyzPoints.push_back(xyz);

        mbr.top = min(query_y, mbr.top);
        mbr.left = min(query_x, mbr.left);
        mbr.right = max(query_x, mbr.right);
        mbr.bottom = max(query_y, mbr.bottom);

        // printf("(%.2f,%.2f,%.2f)", xyz.x, xyz.y, xyz.z);

        if ((count++) % showIntv == 0) {
            Scalar color = CV_RGB(rand() & 64, rand() & 64, rand() & 64);
            sprintf_s(text, 50, "%.2f,%.2f,%.2f", xyz.x, xyz.y, xyz.z);
            putText(temp, text, Point(query_x - 13, query_y - 3),
                    FONT_HERSHEY_SIMPLEX, .3, color);
            circle(temp, queryPoints[matches[i].queryIdx].pt, 2, color, 3);
        }

        // if ((i + 1) % 8 != 0) {
        //     cout << " ";
        // } else
        //     cout << endl;
    }
    // cout << endl;
    imshow("Show XYZ", temp);
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
        srcPoints[i] = queryPoints[matches[i].queryIdx].pt; // 样本图像关键点
        dstPoints[i] = trainPoints[matches[i].trainIdx].pt; // 匹配图像关键点
    }
    // Find homography matrix and get inliers mask
    vector<unsigned char> inliersMask(srcPoints.size());
    homography = findHomography(srcPoints, dstPoints, CV_FM_RANSAC,
            reprojectionThreshold, inliersMask);

    vector<DMatch> inliers;
    vector<DMatch>::iterator iter;
    for (size_t i = 0; i < inliersMask.size(); i++) {
        if (inliersMask[i]) {
            for (iter = inliers.begin(); iter != inliers.end(); ++iter) {
                Point2f diff = srcPoints[i] - queryPoints[iter->queryIdx].pt;
                float dist = abs(diff.x) + abs(diff.y);
                if (dist < match_threshold) //控制匹配点的距离
                    break;
            }
            if (iter != inliers.end())
                continue;
            inliers.push_back(matches[i]);
        }
    }
    matches.swap(inliers);
    return matches.size() > minMatchesNum;
}

void BaseMethods::doSubDiv2D(const vector<KeyPoint>& queryPoints,
        const vector<DMatch>& matches, const PointsMBR& mbr, const Mat& query,
        Subdiv2D& subdiv) {
    // copy for draw image
    Mat img = query.clone();
    Scalar vertex_color(255, 0, 255), edge_color(255, 255, 0);
    for (size_t i = 0; i < matches.size(); ++i) {
        //    drowDelaunayPoint(queryPoints[matches[i].queryIdx].pt, subdiv, img,
        //            edge_color);
        //  // insert delaunay points
        subdiv.insert(queryPoints[matches[i].queryIdx].pt);
    }
    vector<Vec6f> triangles;
    subdiv.getTriangleList(triangles);
    drowDelaunayTriangles(triangles, mbr, img, edge_color);
    drowPointsMBR(mbr, img, Scalar(0, 0, 255));
    imshow("Show Delaunay", img);
}

//插入一个绘制一个，插入完再绘制
void BaseMethods::drowDelaunayPoint(const Point2f point, Subdiv2D& subdiv,
        Mat& img, Scalar color) {
    int e0 = 0, vertex = 0;
    subdiv.locate(point, e0, vertex);
    if (e0 > 0) { // 非虚拟边
        int e = e0;
        do {
            Point2f org, dst;
            if (subdiv.edgeOrg(e, &org) > 0 && subdiv.edgeDst(e, &dst) > 0)
                line(img, org, dst, color, 1, LINE_AA, 0);
            e = subdiv.getEdge(e, Subdiv2D::NEXT_AROUND_ORG);
        } while (e != e0);
    }
    circle(img, point, 2, Scalar(125, 255, 0), FILLED, LINE_8, 0);
}

void BaseMethods::drowDelaunayTriangles(const vector<Vec6f>& triangles,
        const PointsMBR& mbr, Mat& img, Scalar color) {
    vector<Point> pt(3);
    for (size_t i = 0; i < triangles.size(); i++) {
        pt[0] = Point(cvRound(triangles[i][0]), cvRound(triangles[i][1]));
        pt[1] = Point(cvRound(triangles[i][2]), cvRound(triangles[i][3]));
        pt[2] = Point(cvRound(triangles[i][4]), cvRound(triangles[i][5]));
        if (isLineInMBR(mbr, pt[0], pt[1])) {
            line(img, pt[0], pt[1], color, 1, LINE_AA, 0);
        }
        if (isLineInMBR(mbr, pt[1], pt[2])) {
            line(img, pt[1], pt[2], color, 1, LINE_AA, 0);
        }
        if (isLineInMBR(mbr, pt[2], pt[0])) {
            line(img, pt[2], pt[0], color, 1, LINE_AA, 0);
        }
    }
}

bool BaseMethods::isPointInMBR(const PointsMBR& mbr, const Point& point) {
    if (point.x <= mbr.right && point.x >= mbr.left && point.y <= mbr.bottom
            && point.y >= mbr.top)
        return true;
    return false;
}

bool BaseMethods::isLineInMBR(const PointsMBR& mbr, const Point& point1,
        const Point& point2) {
    return isPointInMBR(mbr, point1) && isPointInMBR(mbr, point2);
}

void BaseMethods::drowPointsMBR(const PointsMBR& mbr, Mat& img, Scalar color) {
    vector<Point> mbrPoint(4);
    mbrPoint[0] = Point(mbr.left, mbr.top);
    mbrPoint[1] = Point(mbr.left, mbr.bottom);
    mbrPoint[2] = Point(mbr.right, mbr.bottom);
    mbrPoint[3] = Point(mbr.right, mbr.top);
    line(img, mbrPoint[0], mbrPoint[1], color, 1, LINE_AA, 0);
    line(img, mbrPoint[1], mbrPoint[2], color, 1, LINE_AA, 0);
    line(img, mbrPoint[2], mbrPoint[3], color, 1, LINE_AA, 0);
    line(img, mbrPoint[3], mbrPoint[0], color, 1, LINE_AA, 0);
}

#endif /* JNI_BASEMETHODS_H_ */
