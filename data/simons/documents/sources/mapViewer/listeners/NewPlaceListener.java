package mapViewer.listeners;

import mapPanel.dialogs.*;
import mapViewer.*;
import places.*;

import java.awt.event.*;
import javax.swing.*;

public class NewPlaceListener extends MouseAdapter {

	private Mediator mediator;
	private ControlPanel controlPanel;
	private Categories categories;

	public NewPlaceListener(Mediator mediator, ControlPanel controlPanel, Categories categories) {

		this.mediator = mediator;
		this.controlPanel = controlPanel;
		this.categories = categories;
		
	}

	private PlaceDialog determineDialog(String selectedType, Position position) {

		String selectedCategory = categories.getSelectedCategory();
		
		switch(selectedType) {

			case "Named":
				return new PlaceDialog(selectedCategory, position);
			case "Described":
				return new DescribedPlaceDialog(selectedCategory, position);

		}
		return null;

	}

	@Override
	public void mouseClicked(MouseEvent e) {
		
		String selectedType = controlPanel.getSelectedType();
		PlaceDialog placeDialog = determineDialog(selectedType, new Position(e.getX(), e.getY()));
		JPanel eventFiringPanel = (JPanel) e.getSource(); 
		
		int okOrCancel = JOptionPane.showOptionDialog(eventFiringPanel, 
							placeDialog, 
							"New " + selectedType + " Place", 
							JOptionPane.OK_CANCEL_OPTION, 
							JOptionPane.QUESTION_MESSAGE, 
							null, null, null);

		if(okOrCancel == JOptionPane.OK_OPTION) {
				
			mediator.addPlace(placeDialog.getPlace());
				
		} else {
				
			mediator.removeNewPlaceListener();

		}
	}
	
}
