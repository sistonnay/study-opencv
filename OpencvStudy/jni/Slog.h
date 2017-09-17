/*
 * Slog.h
 *
 *  Created on: 2017��9��17��
 *      Author: ForeverApp
 */

#ifndef JNI_SLOG_H_
#define JNI_SLOG_H_

#ifndef LOG

#ifndef TAG
#define TAG "OpenCV Study Logs: "
#endif

#define LOG(tag, log) \
    printf("%s\n%s: %s\n", TAG, tag, log)
#endif

#endif /* JNI_SLOG_H_ */
