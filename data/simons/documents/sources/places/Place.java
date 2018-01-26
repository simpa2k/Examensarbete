package places;

import javax.swing.*;
import java.awt.*;

public class Place extends JComponent {

	private String category;
	private Position position;
	private String name;
	private boolean marked = false;
	private boolean foldedOut = false;

	public Place(String category, Position position, String name) {

		this.category = category != null ? category : "None";
		this.position = position;
		this.name = name;

		setBounds();

	}

	private void setBounds() {

		setBounds(position.getX() - 10, position.getY() - 20, 20, 20);

	}

	protected Color resolveColor() {

		switch(category) {

			case "Buss":
				return Color.RED;
			case "Tunnelbana":
				return Color.BLUE;
			case "Tåg":
				return Color.GREEN;
			default:
				return Color.BLACK;

		}

	}

	public Color getColor() {

		return resolveColor();

	}

	public String getCategory() {

		return category;

	}

	public Position getPosition() {

		return position;

	}

	public String getName() {
		
		return name;
		
	}
	
	public void makeVisibleAndMarked() {

		setVisible(true);
		setMarked(true);

	}

	public void setMarked(boolean marked) {

		this.marked = marked;
		setBorder();
		repaint();

	}

	public boolean getMarked() {

		return marked;

	}

	public void setFolded(boolean folded) {

		foldedOut = folded;
		
		if(foldedOut) {
			
			getParent().setComponentZOrder(this, 0);

		} else {
			
			setBounds();
			
		}

		repaint();

	}

	public boolean getFolded() {

		return foldedOut;

	}
	
	private void setBorder() {
		
		if(marked) {
			
			setBorder(BorderFactory.createLineBorder(Color.RED));

		} else {
			
			setBorder(null);
			
		}
		
	}

	protected void paintFoldedOut(Graphics g) {
		
		g.setColor(Color.WHITE);
		g.fillRect(0, 0, getWidth(), getHeight());
		
		int nameWidth = g.getFontMetrics().stringWidth(name);
		int nameHeight = g.getFontMetrics().getHeight();

		setBounds(getX(), getY(), nameWidth, getHeight());
		
		g.setColor(Color.BLACK);

		g.drawString(name, 0, nameHeight);
		
	}
	
	@Override
	protected void paintComponent(Graphics g) {
		
		super.paintComponent(g);
		
		if(!foldedOut) {
			
			g.setColor(resolveColor());
			
			int[] xPoints = {getWidth() / 2, 0, getWidth()};
			int[] yPoints = {getHeight(), 0, 0};
			int nPoints = 3;

			g.fillPolygon(xPoints, yPoints, nPoints);
	
		} else {

			paintFoldedOut(g);
			
		}
		
	}

	@Override
	public boolean equals(Object other) {
		
		if(other instanceof Place) {

			Place otherPlace = (Place) other;

			return position.equals(otherPlace.position) && category.equals(otherPlace.category) && name.equals(otherPlace.name);

		}
		return false;

	}

	@Override
	public int hashCode() {
		
		int primeNumber = 37;
		int positiveInteger = 3;
		int sumOfFields = position.hashCode() + category.hashCode() + name.hashCode();

		return primeNumber * positiveInteger + sumOfFields;

	}
	
}
