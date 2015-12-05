package cs224n.deep;

import java.util.*;
import java.io.*;

import org.ejml.simple.SimpleMatrix;

public class NER {

	public static void main(String[] args) throws IOException {
		if (args.length < 2) {
			System.out.println("USAGE: java -cp classes NER ../data/train ../data/dev");
			return;
		}

		// this reads in the train and test datasets
		List<Datum> trainData = FeatureFactory.readTrainData(args[0]);
		List<Datum> testData = FeatureFactory.readTestData(args[1]);

		// read the train and test data
		// TODO: Implement this function (just reads in vocab and word vectors)
		FeatureFactory.initializeVocab("../data/vocab.txt");
		SimpleMatrix allVecs = FeatureFactory.readWordVectors("../data/wordVectors.txt");

		System.out.println(trainData.size());
		// initialize model
		// WindowModel model = new WindowModel(5, 100, 0.001);
		BaselineModel model = new BaselineModel();

		// model.initWeights();

		// TODO: Implement those two functions

		// model.initWeights();
		System.out.println("start  training");
		model.train(trainData);
		System.out.println("start  testing");

		model.test(testData);

	}
}