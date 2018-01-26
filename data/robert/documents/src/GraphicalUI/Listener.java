package GraphicalUI;

import javax.swing.*;
import javax.swing.event.ListSelectionListener;
import java.awt.*;
import java.awt.event.*;
import java.io.File;
import javax.swing.event.*;


/**
 * Created by gentle sir Aaron on 2016-04-12.
 */

public class Listener {
    private GraphicalUI graphicalUI;
    private JFileChooser jfc = new JFileChooser(".");
    private ImagePanel imagePanel;


public Listener (GraphicalUI graphicalUI){
    this.graphicalUI=graphicalUI;
}

    public OpenListener getOpenListener(){
        return new OpenListener();
    }
    public ExitListener getExitListener(){
        return new ExitListener();
    }
    public MarkerListener getMarkerListener(){
        return new MarkerListener();
    }
    public ButtonPressed getButtonPressed(){
        return new ButtonPressed();
    }
    public CategoryListener getCategoryListener() { return new CategoryListener(); }


    class ValueListener implements ActionListener {
        public void actionPerformed(ActionEvent ave) {

            System.out.print("Hej");
        }
    }

    class NameListener implements ActionListener {
        public void actionPerformed(ActionEvent ave) {

            System.out.print("Hej");
        }
    }

    class OpenListener implements ActionListener {
        public void actionPerformed(ActionEvent ave) {
            int svar = jfc.showOpenDialog(graphicalUI);
//            int svar = jfcshowOpenDialog(Visa.this);
            if (svar != JFileChooser.APPROVE_OPTION)
                return;
            File fil = jfc.getSelectedFile();
            String filnamn = fil.getAbsolutePath();
            System.out.println(filnamn);
            imagePanel = new ImagePanel(filnamn);
            graphicalUI.add(imagePanel);
            graphicalUI.pack();
            graphicalUI.validate();
            graphicalUI.repaint();
        }
    }
    class ExitListener implements ActionListener {
        public void actionPerformed(ActionEvent ave) {
            System.exit(0);
        }
    }
    class MarkerListener extends MouseAdapter {
        @Override
        public void mouseClicked(MouseEvent mev) {
        int x = mev.getX();
        int y = mev.getY();
        Marker marker = new Marker(x, y);
        imagePanel.add(marker);
        imagePanel.validate();
        imagePanel.repaint();
//        imagePanel.removeMouseListener(MarkerListener());
            System.out.println(mev.getPoint());

        }
    }
    public class CategoryListener implements ListSelectionListener{
        public void valueChanged(ListSelectionEvent lse) {
            if (!lse.getValueIsAdjusting()) {
                System.out.println(graphicalUI.getJList());
            //    String ordet = graphicalUI.category.getSelectedValue();
//               System.out.println(categories.getText());
            }
        }
    }

    public class ButtonPressed implements ActionListener{
        public void actionPerformed (ActionEvent ave){
            imagePanel.addMouseListener(getMarkerListener());
        }
    }
    }


