package main;


import mapViewer.MapViewerWindow;

public class Main {
	
	public static void main(String[] args) {

		javax.swing.SwingUtilities.invokeLater(new Runnable() {

			public void run() {

				MapViewerWindow mapViewerWindow = new MapViewerWindow();

				mapViewerWindow.setVisible(true);

			}

		});

	}

}
