package mapPanel.dialogs;

import places.*;

import javax.swing.JLabel;
import javax.swing.JOptionPane;
import javax.swing.JPanel;
import javax.swing.JTextField;

public class PlaceDialog extends JPanel {

	private String category;
	private Position position;
	private JTextField nameInput;

	public PlaceDialog(String category, Position position) {

		this.category = category;
		this.position = position;
		
		JLabel name = new JLabel("Name:");
		add(name);

		nameInput = new JTextField(10);
		add(nameInput);

	}

	protected String getCategory() {

		return category;

	}

	protected Position getPosition() {

		return position;

	}
	
	protected String getNameInput() {

		return nameInput.getText();

	}

	public boolean validateInput() {
		
		return (nameInput.getText() != null) && (nameInput.getText().trim().length() != 0);
		
	}
	
	public Place getPlace() {
		
		if(validateInput()) {
			
			return new Place(getCategory(), getPosition(), getNameInput());
			
		} else {
			
			JOptionPane.showMessageDialog(this, "Invalid input", null, JOptionPane.ERROR_MESSAGE);
			return null;
			
		}

	}

}
