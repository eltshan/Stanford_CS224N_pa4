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

		int T = 10;

		FeatureFactory.readWordVectors("../data/wordVectors.txt");
		FeatureFactory.initializeVocab("../data/vocab.txt");
		List<Datum> trainData = FeatureFactory.readTrainData(args[0]);
		List<Datum> testData = FeatureFactory.readTestData(args[1]);
		// BaselineModel model = new BaselineModel();
		WindowModel model = new WindowModel(0, 50, 0.01);
		model.initWeights();
		model.train(trainData);
		// for (int i = 0; i < T; i++) {
		// model.train(trainData);
		//
		// }
		model.test(testData);

		// this reads in the train and test datasets

		// read the train and test data
		// TODO: Implement this function (just reads in vocab and word vectors)
		// SimpleMatrix allVecs =
		// FeatureFactory.readWordVectors("../data/wordVectors.txt");
		// FeatureFactory.initializeVocab("../data/vocab.txt");
		//
		// List<Datum> trainData = FeatureFactory.readTrainData(args[0]);
		// List<Datum> testData = FeatureFactory.readTestData(args[1]);
		//
		// System.out.println(trainData.size());
		// // initialize model
		// // WindowModel model = new WindowModel(5, 100, 0.001);
		// BaselineModel model = new BaselineModel();
		//
		// // model.initWeights();
		//
		// // TODO: Implement those two functions
		//
		// // model.initWeights();
		// System.out.println("start training");
		// model.train(trainData);
		// System.out.println("start testing");
		//
		// model.test(testData);

	}
}