package mapViewer.listeners;

import mapPanel.MapModel;
import mapViewer.Mediator;

import java.awt.event.*;

public class WhatIsHereListener extends MouseAdapter {

	private MapModel mapModel;
	private	Mediator mediator;

	public WhatIsHereListener(MapModel mapModel, Mediator mediator) {

		this.mapModel = mapModel;
		this.mediator = mediator;

	}
	
	@Override
	public void mouseClicked(MouseEvent e) {
		
		mapModel.searchAreaAroundPosition(e.getX(), e.getY());
		mediator.removeWhatIsHereListener();
		
	}

}
