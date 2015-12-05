package cs224n.deep;

import java.util.HashMap;
import java.util.HashSet;
import java.util.List;

public class BaselineModel {

	private HashMap<String, String> map;

	public void train(List<Datum> _trainData) {
		for (Datum datum : _trainData) {
			if (datum.label != "O")
				map.put(datum.word, datum.label);
		}
	}

	public void test(List<Datum> testData) {
		String label = "O";
		for (Datum datum : testData) {
			// if (datum.label != "O")
			// map.add(datum.word);
			label = "O";
			if (map.containsKey(datum.word))
				label = map.get(datum.word);
			System.out.println(datum.word + "\t" + datum.label + "\t" + label);
		}
	}
}
