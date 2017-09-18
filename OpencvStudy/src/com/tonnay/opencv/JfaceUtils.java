package com.tonnay.opencv;

import org.eclipse.swt.SWT;
import org.eclipse.swt.widgets.FileDialog;
import org.eclipse.swt.widgets.Shell;

public class JfaceUtils {
    public static String getFileName(Shell parent) {
        FileDialog dialog = new FileDialog(parent, SWT.OPEN);
        dialog.setFilterPath("");// ����Ĭ�ϵ�·��
        dialog.setText("ѡ���ļ�");// ���öԻ���ı���
        dialog.setFileName("");// ����Ĭ�ϵ��ļ���
        dialog.setFilterNames(new String[] { " (*.bmp)", " (*.jpg)", "(*.*)" });// ������չ��
        dialog.setFilterExtensions(new String[] { "*.bmp", "*.jpg", "*.*" });// �����ļ���չ��
        return dialog.open();
    }

    public static String[] getFileNames(Shell parent) {
        FileDialog dialog = new FileDialog(parent, SWT.OPEN | SWT.MULTI);
        String[] names = dialog.getFileNames();// ��������ѡ����ļ�����������·��
        String path = dialog.getFilterPath(); // ����ѡ���·���������fileNames��Ͽ��Եõ����е��ļ���ȫ·��
        int size = names.length;
        String[] fileNames = new String[size];
        while (--size >= 0) {
            fileNames[size] = path + "/" + names[size];
        }
        return fileNames;
    }
    
    public static String[] getImageDual(Shell parent) {
        FileDialog dialog = new FileDialog(parent, SWT.OPEN);
        dialog.setFilterPath("");// ����Ĭ�ϵ�·��
        dialog.setText("ѡ��ͼƬ1");// ���öԻ���ı���
        dialog.setFileName("");// ����Ĭ�ϵ��ļ���
        dialog.setFilterNames(new String[] { " (*.bmp)", " (*.jpg)", "(*.*)" });// ������չ��
        dialog.setFilterExtensions(new String[] { "*.bmp", "*.jpg", "*.*" });// �����ļ���չ��
        String image1 = dialog.open();
        dialog.setText("ѡ��ͼƬ2");// ���öԻ���ı���
        String image2 = dialog.open();
        return new String[] {image1, image2};
    }
}
