package mapViewer;

import javax.swing.*;
import java.awt.*;

public class Categories extends JPanel {
	
	String[] categories = {"Buss", "Tunnelbana", "Tåg"};
	private JList<String> categoryList; 

	public Categories(Mediator mediator, MapViewerWindow parentFrame) {

		setLayout(new GridBagLayout());
		GridBagConstraints c = new GridBagConstraints();
		
		JPanel centerPanel = new JPanel();

		centerPanel.setLayout(new BorderLayout());
		
		JLabel heading = new JLabel("Categories");
		heading.setHorizontalAlignment(JLabel.CENTER);
		centerPanel.add(heading, BorderLayout.NORTH);
		
		categoryList = new JList<>(categories);
		categoryList.setSelectionMode(ListSelectionModel.SINGLE_SELECTION);
		categoryList.addListSelectionListener(event -> parentFrame.getMapModel().showPlacesByCategory(getSelectedCategory()));
		centerPanel.add(categoryList, BorderLayout.CENTER);

		JButton hideCategory = new JButton("Hide category");
		hideCategory.addActionListener(event -> parentFrame.getMapModel().hidePlacesByCategory(getSelectedCategory())); 
		centerPanel.add(hideCategory, BorderLayout.SOUTH);
		
		add(centerPanel, c);

	}

	public String getSelectedCategory() {

		return categoryList.getSelectedValue();

	}

}
