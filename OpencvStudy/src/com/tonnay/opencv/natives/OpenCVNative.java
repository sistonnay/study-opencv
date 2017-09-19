package com.tonnay.opencv.natives;

public class OpenCVNative {
    
    static {
        System.loadLibrary("opencvstudy");
    }
    
    public native void showImage(String filepath);
    public native void loadImage(String filepath);
    public native void compareDualImages(String img1, String img2);
    
}
