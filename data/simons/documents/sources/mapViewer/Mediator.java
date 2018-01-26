package mapViewer;

import mapPanel.*;
import mapViewer.listeners.NewPlaceListener;
import mapViewer.listeners.WhatIsHereListener;
import places.*;

import javax.swing.*;
import java.awt.*;

public class Mediator {

	private MapModel mapModel;
	private MapViewerWindow mapViewerWindow;
	private MapPanel mapPanel;

	private NewPlaceListener newPlaceListener;
	private WhatIsHereListener whatIsHereListener;

	public Mediator(MapModel mapModel, MapViewerWindow mapViewerWindow, MapPanel mapPanel) {

		this.mapModel = mapModel;
		this.mapViewerWindow = mapViewerWindow;
		this.mapPanel = mapPanel;
	}
 	
	private void setCrosshairCursor(JPanel panel) {

		panel.setCursor(Cursor.getPredefinedCursor(Cursor.CROSSHAIR_CURSOR));

	}

	private void setDefaultCursor(JPanel panel) {

		panel.setCursor(Cursor.getDefaultCursor());

	}

	public void addNewPlaceListener() {

		if( (mapPanel.getImage() != null) && (mapPanel.getMouseListeners().length == 0) ) {
			
			setCrosshairCursor(mapPanel);
			
			newPlaceListener = new NewPlaceListener(this, mapViewerWindow.getControlPanel(), mapViewerWindow.getCategories());
			mapPanel.addMouseListener(newPlaceListener);

		}

	}
	
	public void removeNewPlaceListener() {

		setDefaultCursor(mapPanel);
		mapPanel.removeMouseListener(newPlaceListener);

	}

	public void addWhatIsHereListener() {

		if( (mapPanel != null) && (mapPanel.getMouseListeners().length == 0) ){
			
			setCrosshairCursor(mapPanel);

			whatIsHereListener = new WhatIsHereListener(mapModel, this);
			mapPanel.addMouseListener(whatIsHereListener);

		}

	}

	public void removeWhatIsHereListener() {

		if(mapPanel != null) {

			setDefaultCursor(mapPanel);
			mapPanel.removeMouseListener(whatIsHereListener);

		}

	}

	public void addPlace(Place place) {

		if(place != null) {
			
			mapModel.addNewPlace(place);
			
		}
		
		removeNewPlaceListener();

	}

}
