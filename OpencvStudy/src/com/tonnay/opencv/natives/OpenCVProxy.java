package com.tonnay.opencv.natives;

public class OpenCVProxy {

    private OpenCVProxy() {

    }

    static {
        System.loadLibrary("OpencvStudy");
    }

    public static final OpenCVProxy getInstance() {
        return OpenCVProxyHolder.INSTANCE;
    }

    private static class OpenCVProxyHolder {
        private static final OpenCVProxy INSTANCE = new OpenCVProxy();

    }

    public native void showImage(String image);
    public native void compareImage(String image1, String image2);
}
