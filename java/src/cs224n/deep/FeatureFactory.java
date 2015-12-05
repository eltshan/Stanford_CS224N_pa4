package cs224n.deep;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.*;
import org.ejml.simple.*;

public class FeatureFactory {

	public SimpleMatrix OOV;

	private FeatureFactory() {

	}

	static List<Datum> trainData;

	/** Do not modify this method **/
	public static List<Datum> readTrainData(String filename) throws IOException {
		if (trainData == null)
			trainData = read(filename);
		return trainData;
	}

	static List<Datum> testData;

	/** Do not modify this method **/
	public static List<Datum> readTestData(String filename) throws IOException {
		if (testData == null)
			testData = read(filename);
		return testData;
	}

	private static List<Datum> read(String filename) throws FileNotFoundException, IOException {
		// TODO: you'd want to handle sentence boundaries
		// TODO: test later
		List<Datum> data = new ArrayList<Datum>();
		BufferedReader in = new BufferedReader(new FileReader(filename));
		data.add(new Datum("<s>", "O"));
		for (String line = in.readLine(); line != null; line = in.readLine()) {

			if (line.trim().length() == 0) {
				if (data.get(data.size() - 1).word.equals("<s>"))
					continue;

				data.add(new Datum("</s>", "O"));
				data.add(new Datum("<s>", "O"));
				continue;
			}
			String[] bits = line.split("\\s+");
			String word = bits[0].toLowerCase();
			if (word.equals("-docstart-"))
				continue;
			String label = bits[1];

			Datum datum = new Datum(word, label);
			data.add(datum);
		}
		if (data.get(data.size() - 1).equals("<s>")) {
			data.remove(data.size() - 1);
		}
		if (!data.get(data.size() - 1).equals("</s>")) {
			data.add(new Datum("</s>", "O"));
		}
		in.close();
		return data;
	}

	// Look up table matrix with all word vectors as defined in lecture with
	// dimensionality n x |V|
	static SimpleMatrix allVecs; // access it directly in WindowModel

	public static SimpleMatrix readWordVectors(String vecFilename) throws IOException {
		if (allVecs != null)
			return allVecs;
		File f = new File(vecFilename);
		int numOfRows = 100232;// (int) f.length();
		int numOfCols = 50;
		BufferedReader br = new BufferedReader(new FileReader(vecFilename));

		double[][] tmpMatrixs = new double[numOfRows + 1][numOfCols];
		int row = 0;
		for (String line = br.readLine(); line != null; line = br.readLine()) {

			String[] nums = line.split("\\s+");

			for (int col = 0; col < numOfCols; col++) {
				tmpMatrixs[row][col] = Double.parseDouble(nums[col]);
			}
			row++;
		}

		for (int i = 0; i < 50; i++) {
			tmpMatrixs[numOfRows][i] = new Random().nextDouble();
		}

		br.close();
		allVecs = new SimpleMatrix(tmpMatrixs);
		return allVecs;

	}

	// might be useful for word to number lookups, just access them directly in
	// WindowModel
	public static HashMap<String, Integer> wordToNum = new HashMap<String, Integer>();
	public static HashMap<Integer, String> numToWord = new HashMap<Integer, String>();

	public static HashMap<String, Integer> initializeVocab(String vocabFilename) throws IOException {

		int index = 0;
		BufferedReader br = new BufferedReader(new FileReader(vocabFilename));
		String line;
		while ((line = br.readLine()) != null) {
			wordToNum.put(line, index++);
			numToWord.put(index, line);
		}
		br.close();
		// TODO: create this
		wordToNum.put("OOV", index++);
		numToWord.put(index, "OOV");
		return wordToNum;
	}

}
