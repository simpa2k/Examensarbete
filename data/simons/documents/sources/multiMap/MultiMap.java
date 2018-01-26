package multiMap;

import java.util.*;

public class MultiMap<K, V> {

	HashMap<K, HashSet<V>> map = new HashMap<>();

	public void put(K key, V value) {

		HashSet<V> values = map.get(key);

		if(values == null) {

			values = new HashSet<V>();
			map.put(key, values);

		}
		values.add(value);

	}

	public HashSet<V> get(K key) {

		HashSet<V> values = map.get(key);

		return values;

	}

	public void remove(K key, V value) {

		HashSet<V> values = map.get(key);

		values.remove(value);

		if(values.isEmpty()) {
			
			map.remove(key);

		}

	}

	public void clear() {
		
		map.clear();
		
	}
	
	public boolean isEmpty() {

		return map.isEmpty();

	}

}
