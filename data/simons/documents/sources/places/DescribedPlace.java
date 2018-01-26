package places;

import java.awt.*;

public class DescribedPlace extends Place {

	private String description;

	public DescribedPlace(String category, 
			      Position position, 
			      String name, 
			      String description) {

		super(category, position, name);

		this.description = description;

	}
	
	public String getDescription() {

		return description;

	}

	@Override
	protected void paintFoldedOut(Graphics g) {
		
		super.paintFoldedOut(g);
		
		int descriptionWidth = g.getFontMetrics().stringWidth(description);
		int descriptionHeight = g.getFontMetrics().getHeight();

		setBounds(getX(), getY(), descriptionWidth, descriptionHeight * 3);

		g.drawString(description, 0, descriptionHeight * 2);

	}

	@Override
	public boolean equals(Object other) {
		
		if(other instanceof DescribedPlace) {

			DescribedPlace otherDescribedPlace = (DescribedPlace) other;

			return super.equals(otherDescribedPlace) && description.equals(otherDescribedPlace.description);

		}
		return false;

	}

	@Override
	public int hashCode() {
		
		int primeNumber = 41;
		int sumOfFields = super.hashCode() + description.hashCode();
		
		return primeNumber * sumOfFields;

	}

}
