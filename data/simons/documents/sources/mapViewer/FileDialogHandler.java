package mapViewer;

import java.io.File;
import javax.swing.JFileChooser;
import javax.swing.JOptionPane;
import javax.swing.filechooser.*;
import javax.swing.filechooser.FileFilter;

import mapPanel.MapModel;

public class FileDialogHandler {
	
	private JFileChooser fileChooser = new JFileChooser("./");

	private FileFilter imageFilter = new FileNameExtensionFilter("Images", "jpg", "gif", "png");
	private FileFilter placesFilter = new FileNameExtensionFilter("Places", "places");

	private MapModel mapModel;
	private MapViewerWindow window;

	public FileDialogHandler(MapModel mapModel, MapViewerWindow window) {

		this.mapModel = mapModel;
		this.window = window;

	}
	private File getFile(int okOrCancel) {
	
		if(okOrCancel == JFileChooser.APPROVE_OPTION) {

			return fileChooser.getSelectedFile();

		}
		return null;

	}

	private File openFile() {
		
		int okOrCancel = fileChooser.showOpenDialog(window);
		return getFile(okOrCancel);

	}


	public void openMap() {

		boolean switchingMap = false;
		
		if(mapModel.isChanged()) {
			
			switchingMap = true;
			
			int okOrCancel = JOptionPane.showConfirmDialog(window, 
														"You have unsaved changes, switch map anyway?", 
														"Unsaved changes", 
														JOptionPane.OK_CANCEL_OPTION);
			
			if(okOrCancel != JOptionPane.OK_OPTION) {
				
				return;
			
			}
				
			
		}
		
		fileChooser.setFileFilter(imageFilter);
		File mapFile = openFile();

		if(mapFile != null) {
			
			if(switchingMap) {
				
				mapModel.removeAllPlaces();
				
			}
			
			mapModel.setMapFile(mapFile);

			window.pack();
			window.validate();
			window.makeMenuItemsEnabled();

		}

	}
	
	public void loadPlaces() {
		
		fileChooser.setFileFilter(placesFilter);
		File placesFile = openFile();

		if(placesFile != null) {

			mapModel.loadPlaces(placesFile);

		}

	}

	public void savePlaces() {

		fileChooser.setFileFilter(placesFilter);
		
		int okOrCancel = fileChooser.showSaveDialog(window);		
		File saveFile = getFile(okOrCancel);

		if(saveFile != null) {

			mapModel.savePlaces(saveFile);
		
		}

	}

}
