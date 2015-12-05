package cs224n.deep;

import java.io.IOException;
import java.lang.*;
import java.util.*;

import org.ejml.data.*;
import org.ejml.simple.*;

import java.text.*;

public class WindowModel {

	protected SimpleMatrix L, W, W1, W2, Wout, U, b1, b2, b3;
	protected ArrayList<SimpleMatrix> weight_array, b_array;
	//
	public int num_of_layers, windowSize, wordSize, hiddenSize, hiddenSize_1, hiddenSize_2, K;
	public ArrayList<Integer> size_array;
	private static final String[] LABELS = { "O", "LOC", "MISC", "ORG", "PER" };
	public double lr;
	private static HashMap<String, Integer> labels = new HashMap<String, Integer>();
	double lambda = 1e-4;
	private static final double DIFF_THRESHOLD = 5e-7;

	public WindowModel(int _windowSize, int _hiddenSize, double _lr) {
		windowSize = 3;
		wordSize = 50;
		K = 5;
		hiddenSize = _hiddenSize;
		hiddenSize_1 = _hiddenSize;
		hiddenSize_2 = 10;
		lr = _lr;
	}

	public WindowModel(int _windowSize, int _hiddenSize, double _lr, int num_of_layers, ArrayList<Integer> size_array) {
		windowSize = 3;
		wordSize = 50;
		K = 5;
		hiddenSize = _hiddenSize;
		hiddenSize_1 = _hiddenSize;
		hiddenSize_2 = 10;
		lr = _lr;
	}

	/**
	 * Initializes the weights randomly.
	 */
	public void initWeights_tmp() {
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

	public void initWeights() {
		labels.put("O", 0);
		labels.put("LOC", 1);
		labels.put("MISC", 2);
		labels.put("ORG", 3);
		labels.put("PER", 4);

		W1 = SimpleMatrix.random(hiddenSize_1, windowSize * wordSize, -1, 1, new Random());
		W2 = SimpleMatrix.random(hiddenSize_2, hiddenSize_1, -1, 1, new Random());
		U = SimpleMatrix.random(K, hiddenSize_2, -1, 1, new Random());

		b1 = new SimpleMatrix(hiddenSize_1, 1);
		b2 = new SimpleMatrix(hiddenSize_2, 1);
		b3 = new SimpleMatrix(K, 1);
	}

	/**
	 * Simplest SGD training
	 */
	public void train_tmp(List<Datum> _trainData) {
		double lambda = 1e-4;
		for (int i = 1; i < _trainData.size() - 1; i++) {
			SimpleMatrix x = getTrainingWinodw(_trainData, i);

			SimpleMatrix h = tanh(W.mult(x).plus(b1));
			SimpleMatrix p = softmax(U.mult(h).plus(b2));

			SimpleMatrix y = new SimpleMatrix(K, 1);
			y.set(labels.get(_trainData.get(i).label), 0, 1);
			SimpleMatrix diff = p.minus(y);
			SimpleMatrix db2 = p.minus(y);

			SimpleMatrix dU = diff.mult(h.transpose());
			dU.plus(lambda, U);

			SimpleMatrix db1 = derivativeTanh(h).elementMult(U.transpose().mult(diff));

			SimpleMatrix dW = db1.mult(x.transpose());
			dW.plus(lambda, W);

			SimpleMatrix dx = W.transpose().mult(db1);

			for (int j = 0; j < b2.numRows(); j++) {
				SimpleMatrix b2Prime = b2.copy();
				b2Prime.set(j, 0, b2.get(j, 0) + lambda);
				SimpleMatrix p1 = softmax(U.mult(h).plus(b2Prime));
				b2Prime.set(j, 0, b2.get(j, 0) - lambda);
				SimpleMatrix p2 = softmax(U.mult(h).plus(b2Prime));
				checkGradient(y, p1, p2, db2.get(j, 0));
			}

			// check U
			for (int r = 0; r < U.numRows(); r++) {
				for (int c = 0; c < U.numCols(); c++) {
					SimpleMatrix UPrime = U.copy();
					UPrime.set(r, c, U.get(r, c) + lambda);
					SimpleMatrix p1 = softmax(UPrime.mult(h).plus(b2));
					UPrime.set(r, c, U.get(r, c) - lambda);
					SimpleMatrix p2 = softmax(UPrime.mult(h).plus(b2));
					checkGradient(y, p1, p2, dU.get(r, c));
				}
			}

			// check b1
			for (int r = 0; r < b1.numRows(); r++) {
				SimpleMatrix b1Prime = b1.copy();
				b1Prime.set(r, 0, b1.get(r, 0) + lambda);
				SimpleMatrix hPrime = tanh(W.mult(x).plus(b1Prime));
				SimpleMatrix p1 = softmax(U.mult(hPrime).plus(b2));
				b1Prime.set(r, 0, b1.get(r, 0) - lambda);
				hPrime = tanh(W.mult(x).plus(b1Prime));
				SimpleMatrix p2 = softmax(U.mult(hPrime).plus(b2));
				checkGradient(y, p1, p2, db1.get(r, 0));
			}

			// check W
			for (int r = 0; r < W.numRows(); r++) {
				// check every third element to speed up
				for (int c = 0; c < W.numCols(); c += 3) {
					SimpleMatrix WPrime = W.copy();
					WPrime.set(r, c, W.get(r, c) + lambda);
					SimpleMatrix hPrime = tanh(WPrime.mult(x).plus(b1));
					SimpleMatrix p1 = softmax(U.mult(hPrime).plus(b2));
					WPrime.set(r, c, W.get(r, c) - lambda);
					hPrime = tanh(WPrime.mult(x).plus(b1));
					SimpleMatrix p2 = softmax(U.mult(hPrime).plus(b2));
					checkGradient(y, p1, p2, dW.get(r, c));
				}
			}

			// check x
			// check every other element to speed up
			for (int r = 0; r < x.numRows(); r += 2) {
				SimpleMatrix xPrime = x.copy();
				xPrime.set(r, 0, x.get(r, 0) + lambda);
				SimpleMatrix hPrime = tanh(W.mult(xPrime).plus(b1));
				SimpleMatrix p1 = softmax(U.mult(hPrime).plus(b2));
				xPrime.set(r, 0, x.get(r, 0) - lambda);
				hPrime = tanh(W.mult(xPrime).plus(b1));
				SimpleMatrix p2 = softmax(U.mult(hPrime).plus(b2));
				checkGradient(y, p1, p2, dx.get(r, 0));
			}

			b2 = b2.plus(-lr, db2);
			U = U.plus(-lr, dU);
			b1 = b1.plus(-lr, db1);
			W = W.plus(-lr, dW);
			x = x.plus(-lr, dx);

			/////////
			updateWordVectors(_trainData, i, x);
		}
	}

	public void train(List<Datum> _trainData) {
		SimpleMatrix[] h_array = new SimpleMatrix[num_of_layers];
		SimpleMatrix[] z_array = new SimpleMatrix[num_of_layers];
		SimpleMatrix[] db_array = new SimpleMatrix[num_of_layers];
		SimpleMatrix[] dw_array = new SimpleMatrix[num_of_layers];

		for (int i = 1; i < _trainData.size() - 1; i++) {
			// h_array[0] = getTrainingWinodw(_trainData, i);
			// for (int j = 1; j <= num_of_layers; j++) {
			// z_array[j] = weight_array.get(j).mult(h_array[j -
			// 1]).plus(b_array.get(j));
			// h_array[j] = tanh(z_array[j]);
			// }
			//
			// SimpleMatrix p = softmax(U.mult(h2).plus(b3));
			//
			// for (int j = 1; j <= num_of_layers; j++) {
			// db_array[j] = weight_array.get(j).mult(h_array[j -
			// 1]).plus(b_array.get(j));
			// dw_array[j] = tanh(z_array[j]);
			// }
			SimpleMatrix x = getTrainingWinodw(_trainData, i);
			SimpleMatrix z1 = W1.mult(x).plus(b1);
			SimpleMatrix h1 = tanh(z1);

			SimpleMatrix z2 = W2.mult(h1).plus(b2);
			SimpleMatrix h2 = tanh(z2);

			SimpleMatrix p = softmax(U.mult(h2).plus(b3));

			SimpleMatrix y = new SimpleMatrix(K, 1);
			y.set(labels.get(_trainData.get(i).label), 0, 1);
			SimpleMatrix diff = p.minus(y);
			SimpleMatrix db3 = p.minus(y);

			SimpleMatrix dU = diff.mult(h2.transpose());
			dU.plus(lambda, U);

			SimpleMatrix db2 = derivativeTanh(h2).elementMult(U.transpose().mult(diff));

			SimpleMatrix dW2 = db2.mult(h1.transpose());
			dW2.plus(lambda, W2);

			SimpleMatrix db1 = derivativeTanh(h1).elementMult(W2.transpose().mult(db2));
			SimpleMatrix dW1 = db1.mult(x.transpose());

			SimpleMatrix dx = W1.transpose().mult(db1);
			dW1.plus(lambda, W1);

			b3 = b3.plus(-lr, db3);
			b2 = b2.plus(-lr, db2);
			U = U.plus(-lr, dU);
			b1 = b1.plus(-lr, db1);
			W2 = W2.plus(-lr, dW2);
			W1 = W1.plus(-lr, dW1);
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

			String gold = datum.label;

			// make prediction
			SimpleMatrix x = getTrainingWinodw(testData, i);
			SimpleMatrix scores = score(x);
			String predicted = LABELS[getArgMaxIndex(scores)];

			if (gold.equals(predicted)) {
				correct++;
			} else {
				System.out.println(datum.word + "\t" + gold + "\t" + predicted);
			}
		}
		// System.out.println("done" + correct / testData.size());
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

	private SimpleMatrix score_tmp(SimpleMatrix x) {
		SimpleMatrix h = tanh(W.mult(x).plus(b1));
		return softmax(U.mult(h).plus(b2));
	}

	private void checkGradient(SimpleMatrix y, SimpleMatrix p1, SimpleMatrix p2, double gradient) {
		double slope = (computeLoss(y, p1) - computeLoss(y, p2)) / (2 * lambda);
		if (Math.abs(slope - gradient) > DIFF_THRESHOLD) {
			throw new AssertionError(
					String.format("Gradient check failed: wanted: %.8f, actual: %.8f", slope, gradient));
		}
	}

	private double computeLoss(SimpleMatrix y, SimpleMatrix p) {
		double loss = 0;
		for (int i = 0; i < y.numRows(); i++) {
			loss -= y.get(i, 0) * Math.log(p.get(i, 0));
		}
		loss += computeSquaredMatrixSum(W) * lambda / 2;
		loss += computeSquaredMatrixSum(U) * lambda / 2;
		return loss;
	}

	private double computeSquaredMatrixSum(SimpleMatrix m) {
		return m.elementMult(m).elementSum();
	}

	private SimpleMatrix score(SimpleMatrix x) {
		SimpleMatrix h1 = tanh(W1.mult(x).plus(b1));
		SimpleMatrix h2 = tanh(W2.mult(h1).plus(b2));
		// SimpleMatrix h = tanh(W2.mult(tanh(W1.mult(x).plus(b1)).plus(b2)));
		return softmax(U.mult(h2).plus(b3));
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
		List<Datum> testData = FeatureFactory.readTestData(dataDir + "/dev");

		WindowModel model = new WindowModel(0, 50, 0.01);
		model.initWeights();
		new Random();
		for (int i = 0; i < T; i++) {
			// shuffle list before SGD

			model.train(trainData);

			model.test(testData);
		}

	}
}
