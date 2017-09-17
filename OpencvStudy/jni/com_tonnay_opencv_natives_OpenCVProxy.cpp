/*
 * com_tonnay_opencv_natives_OpenCVProxy.cpp
 *
 *  Created on: 2017Äê9ÔÂ17ÈÕ
 *      Author: ForeverApp
 */
#include "BaseMethods.h"
#include "com_tonnay_opencv_natives_OpenCVProxy.h"

JNIEXPORT void JNICALL Java_com_tonnay_opencv_natives_OpenCVProxy_showImage
  (JNIEnv *env, jobject, jstring jstr)  {
    const char* img = env->GetStringUTFChars(jstr, NULL);
    printf("%s", img);
    BaseMethods::showImage(img);
    env->ReleaseStringUTFChars(jstr, img);
}

JNIEXPORT void JNICALL Java_com_tonnay_opencv_natives_OpenCVProxy_compareImage
  (JNIEnv * env, jobject, jstring jstr1, jstring jstr2) {
    const char* img1 = env->GetStringUTFChars(jstr1, NULL);
    const char* img2 = env->GetStringUTFChars(jstr2, NULL);
    BaseMethods::compareImages(img1, img2);
    env->ReleaseStringUTFChars(jstr1, img1);
    env->ReleaseStringUTFChars(jstr2, img2);
}
