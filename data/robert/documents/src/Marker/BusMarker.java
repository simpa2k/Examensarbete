package Marker;

import java.awt.*;

/**
 * Created by Aaron on 2016-04-14.
 */
public class BusMarker extends Marker {

    public BusMarker(int xes, int yes){
    super(Marker(xes(),yes()));
    }
}

    protected void paintComponent(Graphics g){
        super.paintComponent(g);
        listener.getEListener = new listener.getExitListener();
        g.setColor(Color.BLUE);
        g.fillPolygon(xes, yes, 3);
}
