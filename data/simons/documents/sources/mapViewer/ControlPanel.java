package mapViewer;

import javax.swing.*;
import java.awt.*;

public class ControlPanel extends JPanel {

	Mediator mediator;
	MapViewerWindow parentFrame;
	
	JComboBox<String> namedOrDescribed;
	JTextField searchField;

	public ControlPanel(MapViewerWindow parentFrame, Mediator mediator) {
		
		this.parentFrame = parentFrame;
		this.mediator = mediator;

		setLayout(new GridLayout(0, 1));

		JPanel optionBar = new JPanel();

		JLabel createNew = new JLabel("New:");
		optionBar.add(createNew);

		String[] typesOfPlaces = {"Named", "Described"};
		namedOrDescribed = new JComboBox<>(typesOfPlaces);
		namedOrDescribed.addActionListener(event -> mediator.addNewPlaceListener());
		optionBar.add(namedOrDescribed);

		JTextField searchField = new JTextField("Search", 10);
		optionBar.add(searchField);

		JButton searchButton = new JButton("Search");
		searchButton.addActionListener(event -> parentFrame.getMapModel().markPlacesByName(searchField.getText()));
		optionBar.add(searchButton);

		JButton hideButton = new JButton("Hide");
		hideButton.addActionListener(event -> parentFrame.getMapModel().hideMarkedPlaces());
		optionBar.add(hideButton);

		JButton removeButton = new JButton("Remove");
		removeButton.addActionListener(event -> parentFrame.getMapModel().removeMarkedPlaces());
		optionBar.add(removeButton);

		JButton whatIsHereButton = new JButton("What is here?");
		whatIsHereButton.addActionListener(event -> mediator.addWhatIsHereListener());
		optionBar.add(whatIsHereButton);

		add(optionBar);

	}
	
	public String getSelectedType() {
		
		return (String) namedOrDescribed.getSelectedItem();
		
	}
	
}
