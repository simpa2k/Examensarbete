package Marker;

/**
 * Created by Aaron on 2016-04-14.
 */

import GraphicalUI.Listener;
import GraphicalUI.GraphicalUI;
import javax.swing.*;
import java.awt.*;
import javax.swing.*;
import java.awt.*;
import java.util.Locale;

/**
 * Created by Aaron on 2016-04-12.
 */
public abstract class Marker extends JComponent {
    private GraphicalUI graphicalUI;
//    private GraphicalUI.Listener listener;

    private int [] xes = {0,25,50};
    private int [] yes = {0,50,0};

    public Marker (int x, int y){
        setLayout(null);
        setBounds(x,y,50,50);
        setPreferredSize(new Dimension(50,50));
        setMaximumSize(new Dimension(50,50));
        setMinimumSize(new Dimension(50,50));
//        addActionListener(listener.getButtonPressed());
    }

    protected void paintComponent(Graphics g){
        super.paintComponent(g);
//        listener.getEListener = new listener.getExitListener();
        g.setColor(Color.CYAN);
        g.fillPolygon(xes, yes, 3);
    }

}
