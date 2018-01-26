package mapPanel;

import places.*;

import javax.swing.*;
import java.awt.event.*;

public class PlaceListener extends MouseAdapter {

	private Place place;
	private MapModel mapModel;

	public PlaceListener(Place place, MapModel mapModel) {

		this.place = place;
		this.mapModel = mapModel;

	}

	@Override
	public void mouseClicked(MouseEvent e) {

		if(SwingUtilities.isLeftMouseButton(e)) {
			
			if(place.getMarked()) {

				mapModel.unmarkPlace(place, null);
				
			} else {

				mapModel.markPlace(place);

			}

		} else if(SwingUtilities.isRightMouseButton(e)) {

			place.setFolded(!place.getFolded());

		}

	}

}
