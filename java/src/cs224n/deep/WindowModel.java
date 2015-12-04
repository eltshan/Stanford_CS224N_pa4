package cs224n.deep;

import java.io.IOException;
import java.lang.*;
import java.util.*;

import org.ejml.data.*;
import org.ejml.simple.*;

import java.text.*;

public class WindowModel {

	protected SimpleMatrix L, W, Wout, U, b1, b2;
	//
	public int windowSize, wordSize, hiddenSize, K;

	private static final String[] LABELS = { "O", "LOC", "MISC", "ORG", "PER" };
	public double lr;
	private static HashMap<String, Integer> labels = new HashMap<String, Integer>();

	public WindowModel(int _windowSize, int _hiddenSize, double _lr) {
		windowSize = 3;
		wordSize = 50;
		K = 5;
		hiddenSize = _hiddenSize;
		lr = _lr;
	}

	/**
	 * Initializes the weights randomly.
	 */
	public void initWeights() {
		labels.put("O", 0);
		labels.put("LOC", 1);
		labels.put("MISC", 2);
		labels.put("ORG", 3);
		labels.put("PER", 4);

		W = SimpleMatrix.random(hiddenSize, windowSize * wordSize, -1, 1, new Random());
		U = SimpleMatrix.random(K, hiddenSize, -1, 1, new Random());

		b1 = new SimpleMatrix(hiddenSize, 1);
		b2 = new SimpleMatrix(K, 1);
	}

	/**
	 * Simplest SGD training
	 */
	public void train(List<Datum> _trainData) {
		double lambda = 1e-4;
		for (int i = 1; i < _trainData.size() - 1; i++) {
			SimpleMatrix x = getTrainingWinodw(_trainData, i);

			SimpleMatrix h = tanh(W.mult(x).plus(b1));
			SimpleMatrix p = softmax(U.mult(h).plus(b2));

			SimpleMatrix y = new SimpleMatrix(K, 1);
			y.set(labels.get(_trainData.get(i).label), 0, 1);
			SimpleMatrix diff_p_y = p.minus(y);
			SimpleMatrix db2 = p.minus(y);

			SimpleMatrix dU = diff_p_y.mult(h.transpose());
			dU.plus(lambda, U);

			SimpleMatrix db1 = derivativeTanh(h).elementMult(U.transpose().mult(diff_p_y));

			SimpleMatrix dW = db1.mult(x.transpose());
			dW.plus(lambda, W);

			SimpleMatrix dx = W.transpose().mult(db1);

			b2 = b2.plus(-lr, db2);
			U = U.plus(-lr, dU);
			b1 = b1.plus(-lr, db1);
			W = W.plus(-lr, dW);
			x = x.plus(-lr, dx);

			/////////
			updateWordVectors(_trainData, i, x);
		}
	}

	public void train2(List<Datum> _trainData) {
		double lambda = 1e-4;
		for (int i = 1; i < _trainData.size() - 1; i++) {
			SimpleMatrix x = getTrainingWinodw(_trainData, i);

			SimpleMatrix h = tanh(W.mult(x).plus(b1));
			SimpleMatrix p = softmax(U.mult(h).plus(b2));

			SimpleMatrix y = new SimpleMatrix(K, 1);
			y.set(labels.get(_trainData.get(i).label), 0, 1);
			SimpleMatrix diff_p_y = p.minus(y);
			SimpleMatrix db2 = p.minus(y);

			SimpleMatrix dU = diff_p_y.mult(h.transpose());
			dU.plus(lambda, U);

			SimpleMatrix db1 = derivativeTanh(h).elementMult(U.transpose().mult(diff_p_y));

			SimpleMatrix dW = db1.mult(x.transpose());
			dW.plus(lambda, W);

			SimpleMatrix dx = W.transpose().mult(db1);

//			b3 = b3.plus(-lr, db3);
//			W2 = W2.plus(-lr, d);	
			b2 = b2.plus(-lr, db2);
			U = U.plus(-lr, dU);
			b1 = b1.plus(-lr, db1);
			W = W.plus(-lr, dW);
			x = x.plus(-lr, dx);

			/////////
			updateWordVectors(_trainData, i, x);
		}
	}

	public void test(List<Datum> testData) {
		//
		double correct = 0;
		Datum datum;
		for (int i = 0; i < testData.size(); i++) {
			datum = testData.get(i);
			if (datum.word.equals("<s>") || datum.word.equals("</s>'"))
				continue;

			String gold = datum.label.equals("O") ? "O" : "I-" + datum.label;

			// make prediction
			SimpleMatrix x = getTrainingWinodw(testData, i);
			SimpleMatrix scores = score(x);
			String predicted = LABELS[getArgMaxIndex(scores)];
			if (!predicted.equals("O")) {
				predicted = "I-" + predicted;
			}
			if (gold.equals(predicted)) {
				correct++;
			}
		}
		System.out.println("done" + correct / testData.size());
		// out.close();
	}

	private int getArgMaxIndex(SimpleMatrix scores) {
		int idx = 0;
		for (int i = 1; i < scores.numRows(); i++) {
			if (scores.get(i, 0) > scores.get(idx, 0))
				idx = i;
		}
		return idx;
	}

	private SimpleMatrix score(SimpleMatrix x) {
		SimpleMatrix h = tanh(W.mult(x).plus(b1));
		return softmax(U.mult(h).plus(b2));
	}

	private SimpleMatrix getTrainingWinodw(List<Datum> _trainData, int index) {
		SimpleMatrix result = new SimpleMatrix(windowSize * wordSize, 1);
		for (int i = index - windowSize / 2; i <= Math.min(index + windowSize / 2, _trainData.size() - 1); i++) {

			SimpleMatrix wordvec;
			int row = -1;
			if (FeatureFactory.wordToNum.containsKey(_trainData.get(i).word))
				row = FeatureFactory.wordToNum.get(_trainData.get(i).word);
			else
				row = FeatureFactory.wordToNum.get("OOV");

			wordvec = FeatureFactory.allVecs.extractVector(true, row);

			result.insertIntoThis((i - (index - windowSize / 2)) * wordSize, 0, wordvec.transpose());
		}

		return result;
	}

	private SimpleMatrix tanh(SimpleMatrix x) {
		for (int i = 0; i < x.numRows(); i++) {
			for (int j = 0; j < x.numCols(); j++) {
				x.set(i, j, Math.tanh(x.get(i, j)));
			}
		}
		return x;
	}

	private SimpleMatrix softmax(SimpleMatrix x) {
		SimpleMatrix result = new SimpleMatrix(K, 1);
		double sum = 0;
		for (int i = 0; i < K; i++) {
			sum += Math.exp(x.get(i, 0));
		}
		for (int i = 0; i < K; i++) {
			result.set(i, 0, Math.exp(x.get(i, 0)) / sum);
		}
		return result;
	}

	private SimpleMatrix derivativeTanh(SimpleMatrix h) {
		// SimpleMatrix result = h.copy();
		SimpleMatrix result = new SimpleMatrix(h);
		for (int r = 0; r < result.numRows(); r++) {
			double tanh = result.get(r, 0);
			result.set(r, 0, 1 - tanh * tanh);
		}
		return result;
	}

	private void updateWordVectors(List<Datum> _trainData, int index, SimpleMatrix x) {
		for (int i = index - windowSize / 2; i <= index + windowSize / 2; i++) {
			int num = 0;
			if (FeatureFactory.wordToNum.containsKey(_trainData.get(i).word))
				num = FeatureFactory.wordToNum.get((_trainData.get(i).word));
			else
				num = FeatureFactory.wordToNum.get(("OOV"));
			FeatureFactory.allVecs.insertIntoThis(num, 0, x.extractMatrix((i - (index - windowSize / 2)) * wordSize,
					((i - (index - windowSize / 2)) + 1) * wordSize, 0, 1).transpose());
		}
	}

	public static void main(String[] args) throws IOException {
		int T = 10;
		String dataDir = "C:/Users/Eltshan/git/Stanford_CS224N_pa4/data";

		FeatureFactory.readWordVectors(dataDir + "/wordVectors.txt");
		FeatureFactory.initializeVocab(dataDir + "/vocab.txt");
		List<Datum> trainData = FeatureFactory.readTrainData(dataDir + "/train");
		System.out.println(dataDir + "/test");
		FeatureFactory.readTestData(dataDir + "/dev");

		WindowModel model = new WindowModel(0, 50, 0.01);
		model.initWeights();
		new Random();
		for (int i = 0; i < T; i++) {
			System.out.println("\nTraining round " + (i + 1));
			// shuffle list before SGD

			model.train(trainData);

			model.test(trainData);
		}

	}
}
