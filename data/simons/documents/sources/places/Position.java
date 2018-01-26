package places;

public class Position {

	private int x;
	private int y;

	public Position(int x, int y) {

		this.x = x;
		this.y = y;

	}

	public int getX() {

		return x;

	}

	public int getY() {

		return y;

	}

	@Override
	public boolean equals(Object other) {

		if(other instanceof Position) {
			
			Position position = (Position) other;

			return (position.getX() == x) && (position.getY() == y);

		}
		return false;

	}

	@Override
	public int hashCode() {
		
		int primeNumber = 37;
		int positiveInteger = 6;
		int sumOfFields = x + y;
		
		return primeNumber * positiveInteger + sumOfFields;

	}

	@Override
	public String toString() {
		
		return x + "," + y;

	}
}
