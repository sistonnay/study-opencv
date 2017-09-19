/*
 * tonnay_opencv_OpenCVNative.cpp
 *
 *  Created on: 2017Äê9ÔÂ13ÈÕ
 *      Author: hetao
 */
#include "BaseMethods.h"
#include "com_tonnay_opencv_natives_OpenCVNative.h"

JNIEXPORT void JNICALL Java_com_tonnay_opencv_natives_OpenCVNative_showImage(
        JNIEnv * env, jobject, jstring jstr) {
    const char* str = env->GetStringUTFChars(jstr, NULL);
    if (str == NULL)
        return;
    BaseMethods::showImage(str);
    env->ReleaseStringUTFChars(jstr, str);
}

JNIEXPORT void JNICALL Java_com_tonnay_opencv_natives_OpenCVNative_loadImage(
        JNIEnv *, jobject, jstring) {

}

JNIEXPORT void JNICALL Java_com_tonnay_opencv_natives_OpenCVNative_compareDualImages(
        JNIEnv * env, jobject, jstring jstr1, jstring jstr2) {
    const char* str1 = env->GetStringUTFChars(jstr1, NULL);
    const char* str2 = env->GetStringUTFChars(jstr2, NULL);
    if (str1 == NULL || str2 == NULL)
        return;
    BaseMethods::compareImages(str1, str2);
    env->ReleaseStringUTFChars(jstr1, str1);
    env->ReleaseStringUTFChars(jstr2, str2);
}
