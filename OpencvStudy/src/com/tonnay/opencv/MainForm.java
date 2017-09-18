package com.tonnay.opencv;

import org.eclipse.jface.action.MenuManager;
import org.eclipse.jface.action.StatusLineManager;
import org.eclipse.jface.action.ToolBarManager;
import org.eclipse.jface.window.ApplicationWindow;
import org.eclipse.swt.SWT;
import org.eclipse.swt.graphics.Point;
import org.eclipse.swt.widgets.Composite;
import org.eclipse.swt.widgets.Control;
import org.eclipse.swt.widgets.Display;
import org.eclipse.swt.widgets.Shell;
import org.eclipse.swt.widgets.Button;
import org.eclipse.swt.events.SelectionAdapter;
import org.eclipse.swt.events.SelectionEvent;

public class MainForm extends ApplicationWindow {

    /**
     * Create the application window.
     */
    public MainForm() {
        super(null);
        createActions();
        addToolBar(SWT.FLAT | SWT.WRAP);
        addMenuBar();
        addStatusLine();
    }

    /**
     * Create contents of the application window.
     * @param parent
     */
    @Override
    protected Control createContents(Composite parent) {
        Composite container = new Composite(parent, SWT.NONE);
        
        {
            Button btnShowimage = new Button(container, SWT.NONE);
            btnShowimage.addSelectionListener(new SelectionAdapter() {
                @Override
                public void widgetSelected(SelectionEvent e) {
                    String filepath = JfaceUtils.getFileName(getShell());
                    System.out.println(filepath);
                    OpenCVTest.getInstance().testImgShow(filepath);
                }
            });
            btnShowimage.setBounds(10, 10, 120, 32);
            btnShowimage.setText("ShowImage");
        }
        
        {
            Button btnLoaddualimg = new Button(container, SWT.NONE);
            btnLoaddualimg.addSelectionListener(new SelectionAdapter() {
                @Override
                public void widgetSelected(SelectionEvent e) {
                    String[] dualImg = JfaceUtils.getImageDual(getShell());
                    System.out.println(dualImg[0] + "\n" + dualImg[1]);
                    OpenCVTest.getInstance().compareDualImages(dualImg[0] , dualImg[1]);
                }
            });
            btnLoaddualimg.setBounds(10, 48, 120, 32);
            btnLoaddualimg.setText("LoadDualImage");
        }

        return container;
    }

    /**
     * Create the actions.
     */
    private void createActions() {
        // Create the actions
    }

    /**
     * Create the menu manager.
     * @return the menu manager
     */
    @Override
    protected MenuManager createMenuManager() {
        MenuManager menuManager = new MenuManager("menu");
        return menuManager;
    }

    /**
     * Create the toolbar manager.
     * @return the toolbar manager
     */
    @Override
    protected ToolBarManager createToolBarManager(int style) {
        ToolBarManager toolBarManager = new ToolBarManager(style);
        return toolBarManager;
    }

    /**
     * Create the status line manager.
     * @return the status line manager
     */
    @Override
    protected StatusLineManager createStatusLineManager() {
        StatusLineManager statusLineManager = new StatusLineManager();
        return statusLineManager;
    }

    /**
     * Launch the application.
     * @param args
     */
    public static void main(String args[]) {
        try {
            MainForm window = new MainForm();
            window.setBlockOnOpen(true);
            window.open();
            Display.getCurrent().dispose();
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    /**
     * Configure the shell.
     * @param newShell
     */
    @Override
    protected void configureShell(Shell newShell) {
        super.configureShell(newShell);
        newShell.setText("New Application");
    }

    /**
     * Return the initial size of the window.
     */
    @Override
    protected Point getInitialSize() {
        return new Point(450, 300);
    }
}
