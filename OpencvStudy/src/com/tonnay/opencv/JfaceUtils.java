package com.tonnay.opencv;

import org.eclipse.swt.SWT;
import org.eclipse.swt.widgets.FileDialog;
import org.eclipse.swt.widgets.Shell;

public class JfaceUtils {
    public static String getFileName(Shell parent) {
        FileDialog dialog = new FileDialog(parent, SWT.OPEN);
        dialog.setFilterPath("");// 设置默认的路径
        dialog.setText("选择文件");// 设置对话框的标题
        dialog.setFileName("");// 设置默认的文件名
        dialog.setFilterNames(new String[] { " (*.bmp)", " (*.jpg)", "(*.*)" });// 设置扩展名
        dialog.setFilterExtensions(new String[] { "*.bmp", "*.jpg", "*.*" });// 设置文件扩展名
        return dialog.open();
    }

    public static String[] getFileNames(Shell parent) {
        FileDialog dialog = new FileDialog(parent, SWT.OPEN | SWT.MULTI);
        String[] names = dialog.getFileNames();// 返回所有选择的文件名，不包括路径
        String path = dialog.getFilterPath(); // 返回选择的路径，这个和fileNames配合可以得到所有的文件的全路径
        int size = names.length;
        String[] fileNames = new String[size];
        while (--size >= 0) {
            fileNames[size] = path + "/" + names[size];
        }
        return fileNames;
    }
    
    public static String[] getImageDual(Shell parent) {
        FileDialog dialog = new FileDialog(parent, SWT.OPEN);
        dialog.setFilterPath("");// 设置默认的路径
        dialog.setText("选择图片1");// 设置对话框的标题
        dialog.setFileName("");// 设置默认的文件名
        dialog.setFilterNames(new String[] { " (*.bmp)", " (*.jpg)", "(*.*)" });// 设置扩展名
        dialog.setFilterExtensions(new String[] { "*.bmp", "*.jpg", "*.*" });// 设置文件扩展名
        String image1 = dialog.open();
        dialog.setText("选择图片2");// 设置对话框的标题
        String image2 = dialog.open();
        return new String[] {image1, image2};
    }
}
