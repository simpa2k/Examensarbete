package mapPanel.dialogs;

import places.*;
import javax.swing.*;

public class DescribedPlaceDialog extends PlaceDialog {

	JTextField descriptionInput;

	public DescribedPlaceDialog(String category, Position position) {

		super(category, position);

		JLabel description = new JLabel("Description:");
		add(description);

		descriptionInput = new JTextField(10);
		add(descriptionInput);

	}

	protected String getDescriptionInput() {

		return descriptionInput.getText();

	}
	
	@Override
	public boolean validateInput() {
		
		boolean validName = super.validateInput();
		
		return validName && (descriptionInput.getText() != null) && (descriptionInput.getText().trim().length() != 0);
		
	}
	
	@Override
	public DescribedPlace getPlace() {
		
		if(validateInput()) {
		
			return new DescribedPlace(getCategory(), getPosition(), getNameInput(), getDescriptionInput());
			
		} else {
			
			JOptionPane.showMessageDialog(this, "Invalid input", null, JOptionPane.ERROR_MESSAGE);
			return null;
			
		}

	}
}
