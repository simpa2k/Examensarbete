package mapPanel;

import places.*;
import multiMap.MultiMap;

import java.io.*;
import java.util.*;
import javax.swing.JOptionPane;

public class MapModel {

	private MapPanel view;

	private HashMap<Position, Place> placesByPosition = new HashMap<>();
	private MultiMap<String, Place> placesByName = new MultiMap<>();
	private MultiMap<String, Place> placesByCategory = new MultiMap<>();
	private HashSet<Place> markedPlaces = new HashSet<>();

	private boolean changed = false;

	public boolean isChanged() {

		return changed;

	}
	
	protected void setView(MapPanel view) {

		this.view = view;

	}

	public void setMapFile(File mapFile) {
			
		view.setMap(mapFile);

	}
	
	private void addPlace(Place place) {

		place.addMouseListener(new PlaceListener(place, this));
		
		placesByPosition.put(place.getPosition(), place);
		placesByCategory.put(place.getCategory(), place);
		placesByName.put(place.getName(), place);


		view.drawPlace(place);

	}

	public void addNewPlace(Place place) {

		addPlace(place);	
		changed = true;

	}

	private void createPlace(String type, String category, int xPosition, int yPosition, String name, String description) {

		Place place = null;

		switch(type) {
			
			case("Named"):
				place = new Place(category, 
								  new Position(xPosition, yPosition), 
								  name);
			break;
			
			case("Described"):
				place = new DescribedPlace(category, 
										   new Position(xPosition, yPosition), 
										   name,
										   description);
			break;
		}
		
		if(place != null) {
			
			addPlace(place);
		
		}
	}
	
	private void parsePlaceLine(String[] properties) {

		String type = properties[0];
		String category = properties[1];
		int xPosition = Integer.parseInt(properties[2]);
		int yPosition = Integer.parseInt(properties[3]);
		String name = properties[4];
		String description = properties.length == 6 ? properties[5] : null;

		createPlace(type, category, xPosition, yPosition, name, description);

	}

	public void loadPlaces(File placesFile) {
		
		try(BufferedReader reader = new BufferedReader(new FileReader(placesFile))) {
			
			String line;

			while( (line = reader.readLine()) != null ) {

				String[] properties = line.split(",");
				
				parsePlaceLine(properties);
					
			}

		} catch(FileNotFoundException e) {
			
			JOptionPane.showMessageDialog(view, "Places file not found.", "File not found", JOptionPane.ERROR_MESSAGE);
			
		} catch(IOException e) {

			JOptionPane.showMessageDialog(view, e.getMessage(), "Error", JOptionPane.ERROR_MESSAGE);

		} 

	}
	
	private String getPlaceAsLine(Place place) {
		
		Position position = place.getPosition();
		String line = place.getCategory() + "," + position.getX() + "," + position.getY() + "," + place.getName();
		
		if(place instanceof DescribedPlace) {
			
			DescribedPlace describedPlace = (DescribedPlace) place;
			line = "Described," + line + "," + describedPlace.getDescription();
			
		} else {
			
			line = "Named," + line; 
			
		} 
		
		return line;
			
	}
	
	public void savePlaces(File saveFile) {
		
		if(!(placesByPosition.isEmpty())) {
			
			try(PrintWriter printWriter = new PrintWriter(new FileWriter(saveFile))) {
			
				placesByPosition.forEach( (position, place) -> {		
	
						printWriter.println(getPlaceAsLine(place));

				});
				
			} catch(FileNotFoundException e) {
				
				JOptionPane.showMessageDialog(view, "Could not find file for writing", "File not found", JOptionPane.ERROR_MESSAGE);
				
			} catch(IOException e) {
				
				JOptionPane.showMessageDialog(view, e.getMessage(), "Error", JOptionPane.ERROR_MESSAGE);
				
			}
			changed = false;

		}
	}
	
	public void searchAreaAroundPosition(int x, int y) {

		for(int offsetX = -10; offsetX < 11; offsetX++) {
			
			for(int offsetY = -10; offsetY < 11; offsetY++) {
				
				Position position = new Position(x + offsetX, y + offsetY);
				
				if(placesByPosition.containsKey(position)) {
					
					placesByPosition.get(position).setVisible(true);
					
				}
				
			}
				
		}

	}
	
	public void markPlace(Place place) {
	
		place.makeVisibleAndMarked();
		markedPlaces.add(place);

	}
	
	public void unmarkPlace(Place place, Iterator<Place> iterator) {
		
		place.setMarked(false);
		
		if(iterator == null) {
			
			markedPlaces.remove(place);
			
		} else {
			
			iterator.remove();
			
		}
		

	}
	
	private void removePlacesAlreadyMarked(HashSet<Place> placesToBeMarked) {
		
		Iterator<Place> iterator =  markedPlaces.iterator();

		while(iterator.hasNext()) {

			Place place = iterator.next();

			if(placesToBeMarked.contains(place)) {

				placesToBeMarked.remove(place);

			} else {

				unmarkPlace(place, iterator);

			}
		}
		
	}
	
	public void markPlacesByName(String name) {

		if(!(placesByName.isEmpty())) {		

			HashSet<Place> placesToBeMarked = placesByName.get(name);
			
			if(placesToBeMarked == null) {
				
				JOptionPane.showMessageDialog(view, 
										"There are no places with that name.", 
										"Name not found", 
										JOptionPane.ERROR_MESSAGE);
				return;
				
			}

			if(!(markedPlaces.isEmpty())) {

					removePlacesAlreadyMarked(placesToBeMarked);
					
					if(placesToBeMarked.isEmpty()) {

						return;
						
					}

			}

			for(Place place : placesToBeMarked) {

				markPlace(place);

			}
				
		}

	}
	
	public void showPlacesByCategory(String category) {
		
		HashSet<Place> places = placesByCategory.get(category);
		
		if(places != null) {
			
			places.forEach(place -> {
			
				place.setVisible(true);
				
			});
		
		}
		
	}
	
	private void hidePlace(Place place, Iterator<Place> iterator) {
		
		place.setVisible(false);
		unmarkPlace(place, iterator);
		
	}
	
	public void hideMarkedPlaces() {

		Iterator<Place> iterator = markedPlaces.iterator();
		
		while(iterator.hasNext()) {
			
			Place place = iterator.next();
			
			hidePlace(place, iterator);
			
		}
		
	}
	
	public void hidePlacesByCategory(String category) {
		
		HashSet<Place> places = placesByCategory.get(category);
		
		if(places != null) {
			
			places.forEach(place -> hidePlace(place, null));
		
		}

		
	}
	
	public void removeMarkedPlaces() {

		Iterator<Place> iterator = markedPlaces.iterator();

		while(iterator.hasNext()) {

			Place place = (Place) iterator.next();

			String name = place.getName();
			placesByName.remove(name, place);
				
			Position position = place.getPosition();
			placesByPosition.remove(position);
			
			view.remove(place);
			view.repaint();
			
			iterator.remove();
		}

		changed = true;

	}
	
	public void removeAllPlaces() {
		
		placesByPosition.clear();
		placesByName.clear();
		placesByCategory.clear();
		markedPlaces.clear();
		
		view.removeAll();
		changed = false;
		
	}
}
