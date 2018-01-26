package mapViewer.listeners;

import mapPanel.MapModel;

import javax.swing.*;
import java.awt.Component;
import java.awt.event.*;

public class ExitListener extends WindowAdapter implements ActionListener {

	private MapModel mapModel;
	private Component eventFiringComponent;

	public ExitListener(MapModel mapModel, Component eventFiringComponent) {

		this.mapModel = mapModel;
		this.eventFiringComponent = eventFiringComponent;

	}
	
	public void actionPerformed(ActionEvent e) {
		
		checkModelStatus();

	}

	public void windowClosing(WindowEvent e) {
		
		checkModelStatus();

	}

	private void checkModelStatus() {

		if(mapModel.isChanged()) {

			int okOrCancel = JOptionPane.showConfirmDialog(eventFiringComponent, "You have unsaved changes, exit anyway?",
								       "Unsaved changes", JOptionPane.OK_CANCEL_OPTION);

			if(okOrCancel == JOptionPane.OK_OPTION) {

				System.exit(0);

			}

		} else {

			System.exit(0);

		}	

	}

}
