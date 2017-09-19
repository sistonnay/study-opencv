package com.tonnay.opencv;

import com.tonnay.opencv.natives.OpenCVNative;

public class OpenCVTest {
    
    private static OpenCVNative mOpenCVNative;
    private static volatile OpenCVTest mOpenCVTest;
    
    private OpenCVTest() {
        mOpenCVNative = new OpenCVNative();
    }
    
    public static OpenCVTest getInstance() {
        if (mOpenCVTest == null) {
            synchronized (OpenCVTest.class) {
                if (mOpenCVTest == null) {
                    mOpenCVTest = new OpenCVTest();
                }
            }
            
        }
        return mOpenCVTest;
    }
    
    public void testImgShow(String filepath) {
        mOpenCVNative.showImage(filepath);
    }

    public void testLoadImage(String filepath) {
        mOpenCVNative.loadImage(filepath);
    }
    
    public void compareDualImages(String ImgFile1, String ImgFile2) {
        mOpenCVNative.compareDualImages(ImgFile1, ImgFile2);
    }
}
