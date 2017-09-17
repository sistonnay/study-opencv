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

class BaseMethods {
public:
    BaseMethods();
    virtual ~BaseMethods();

    static int showImage(const char* img);
    static int compareImages(const char* img1, const char* img2);
    static void matchFeatures(Mat& query, Mat& train, vector<DMatch>& matches);
};

struct SurfParams {
    Mat * src = NULL;
    Mat * dst = NULL;
    vector<KeyPoint> * keypoints = NULL;
    bool flag = false;
    //引用在定义的时候必须被初始化，所以此处只能用指针
    SurfParams(Mat* src, Mat* dst, vector<KeyPoint>* keypoints) {
        this->src = src;
        this->dst = dst;
        this->keypoints = keypoints;
    }
};

void * surfThread(void * param) {
    // time_t t =time(NULL);
    // time(NULL) 这句返回的只是一个时间cuo, 精度秒
    struct timeval start, end;
    gettimeofday( &start, NULL ); // Linux 下

    struct SurfParams * temp;
    temp = (SurfParams *) param;
    Ptr<Feature2D> surf = SURF::create();
    surf->detectAndCompute(*temp->src, noArray(), *temp->keypoints, *temp->dst);
    temp->flag = true;

    gettimeofday( &end, NULL );
    int coast = 1000000 * ( end.tv_sec - start.tv_sec ) + end.tv_usec - start.tv_usec;
    char deta[32];
    sprintf(deta, "Total coast of SURF: %d ms.", coast / 1000);
    LOG(TAG_BAGE, deta);
    return NULL;
}

int BaseMethods::showImage(const char* img) {
    Mat src = imread(img, IMREAD_COLOR);
    if (!src.data) {
        LOG(TAG_BAGE, "No data!--Exiting the program!");
        return -1;
    }

    Mat dst;
    Size size_src = src.size();
    int width_dst = 800;
    int height_dst = size_src.height * width_dst / size_src.width;
    Size size_dst(width_dst, height_dst);
    resize(src, dst, size_dst, 0, 0, INTER_LINEAR);

    namedWindow("Show Image", CV_WINDOW_AUTOSIZE);
    imshow("Show Image", dst);
    waitKey(0);

    return 1;
}

int BaseMethods::compareImages(const char* img1, const char* img2) {

    Mat src_1 = imread(img1);
    Mat src_2 = imread(img2);

    Size size_src1 = src_1.size();
    Size size_src2 = src_2.size();

    Mat rsrc_1, rsrc_2;
    resize(src_1, rsrc_1, size_src1 / 8, 0, 0, INTER_LINEAR);
    resize(src_2, rsrc_2, size_src2 / 8, 0, 0, INTER_LINEAR);

    struct timeval start, end;
    gettimeofday( &start, NULL ); // Linux 下

    Mat dest_1, dest_2;
    vector<KeyPoint> keypoints_1, keypoints_2;

    SurfParams surfParam1(&rsrc_1, &dest_1, &keypoints_1);
    SurfParams surfParam2(&rsrc_2, &dest_2, &keypoints_2);

    pthread_t th1, th2;
    pthread_create(&th1, NULL, surfThread, &surfParam1);
    pthread_create(&th2, NULL, surfThread, &surfParam2);
    while (!(surfParam1.flag && surfParam2.flag));

    vector<DMatch> matches;
    matchFeatures(dest_1, dest_2, matches);

    gettimeofday( &end, NULL );
    int coast = 1000000 * ( end.tv_sec - start.tv_sec ) + end.tv_usec - start.tv_usec;
    char deta[32];
    sprintf(deta, "Total coast of Match: %d ms.", coast / 1000);
    LOG(TAG_BAGE, deta);

    Mat match_keypoints;
    drawMatches(rsrc_1, keypoints_1, rsrc_2, keypoints_2, matches,
            match_keypoints, Scalar::all(-1), Scalar::all(-1), vector<char>(),
            DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);

    imshow("Matching Result", match_keypoints);
    //imwrite("match_result.png", match_keypoints);
    waitKey(0);

    return 1;
}

void BaseMethods::matchFeatures(Mat& query, Mat& train,
        vector<DMatch>& matches) {
    vector<vector<DMatch>> knn_matches;
    BFMatcher matcher(NORM_L2);
    matcher.knnMatch(query, train, knn_matches, 2);

    //获取满足Ratio Test的最小匹配的距离
    float min_dist = FLT_MAX;
    for (unsigned r = 0; r < knn_matches.size(); ++r) {
        //Ratio Test
        if (knn_matches[r][0].distance > 0.6 * knn_matches[r][1].distance)
            continue;
        float dist = knn_matches[r][0].distance;
        if (dist < min_dist)
            min_dist = dist;
    }

    matches.clear();
    for (size_t r = 0; r < knn_matches.size(); ++r) {
        //排除不满足Ratio Test的点和匹配距离过大的点
        if (knn_matches[r][0].distance > 0.6 * knn_matches[r][1].distance
                || knn_matches[r][0].distance > 5 * max(min_dist, 10.0f))
            continue;
        //保存匹配点
        matches.push_back(knn_matches[r][0]);
    }
}

#endif /* JNI_BASEMETHODS_H_ */
