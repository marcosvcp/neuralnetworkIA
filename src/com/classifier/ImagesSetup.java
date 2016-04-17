package com.classifier;

import weka.classifiers.Evaluation;
import weka.core.FastVector;
import weka.core.Instances;

import javax.imageio.ImageIO;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.io.File;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashSet;
import java.util.List;

public class ImagesSetup {

    private static final String DIGITOS = "digitos";
    private static final String LETRAS = "letras";
    private static final String DIGITOS_LETRAS = "digitos_letras";
    private static final String SEM_CARACTERES = "sem_caracteres";


    public static void main(String[] a) throws Exception {
        boolean verbose = Boolean.parseBoolean(a[0]);
        String pathTrain = a[1];
        String pathTest = a[2];

        FastVector classes = new FastVector(4);
        classes.addElement(DIGITOS);
        classes.addElement(LETRAS);
        classes.addElement(DIGITOS_LETRAS);
        classes.addElement(SEM_CARACTERES);
        NeuralNetworkClassifier NNCTrain = new NeuralNetworkClassifier(classes);
        String folderDigits = pathTrain + "/digitos";
        String folderLetters = pathTrain + "/letras";
        String folderBoth = pathTrain + "/digitos_letras";
        String nothing = pathTrain + "/sem_caracteres";

        NNCTrain.buildSet(folderDigits, DIGITOS, null);
        NNCTrain.buildSet(folderLetters, LETRAS, null);
        NNCTrain.buildSet(folderBoth, DIGITOS_LETRAS, null);
        NNCTrain.buildSet(nothing, SEM_CARACTERES, null);
        NNCTrain.buildClassifier();

        Evaluation eTest = new Evaluation(NNCTrain.getSet());
        ClassifierSetBuilder testBuilder = new ClassifierSetBuilder(classes);
        String folderTestLetters = pathTest + "/letras";
        String folderTestDigits = pathTest + "/digitos";
        String folderTestBoth = pathTest + "/digitos_letras";
        String nothingTest = pathTest + "/sem_caracteres";
        List<String> paths = new ArrayList<String>();
        testBuilder.buildSet(folderTestLetters, LETRAS, paths);
        testBuilder.buildSet(folderTestDigits, DIGITOS, paths);
        testBuilder.buildSet(folderTestBoth, DIGITOS_LETRAS, paths);
        testBuilder.buildSet(nothingTest, SEM_CARACTERES, paths);

        Instances testSet = testBuilder.getSet();
        eTest.evaluateModel(NNCTrain.getClassifier(), testSet);

        getStatistics(verbose, NNCTrain, eTest, paths, testSet);
    }

    private static void getStatistics(boolean verbose, NeuralNetworkClassifier NNCTrain, Evaluation eTest, List<String> paths, Instances testSet) throws Exception {
        for (int i = 0; i < testSet.numInstances(); i++) {
            double pred = NNCTrain.getClassifier().classifyInstance(testSet.instance(i));
            System.out.println("ID: " + paths.get(i));
            String actual = testSet.classAttribute().value((int) testSet.instance(i).classValue());
            String predicted = testSet.classAttribute().value((int) pred);
            System.out.println("actual: " + actual);
            System.out.println("predicted: " + predicted);
            System.out.println("");
            if (actual.equals(predicted)) {
                System.out.println("SUCCESS");
            } else {
                System.out.println("FAILURE");
            }
        }

        if (verbose) {
            System.out.println(eTest.toSummaryString(true));
            System.out.println(eTest.toClassDetailsString());
        }

        System.out.println("precision: " + eTest.weightedPrecision());
        System.out.println("recall: " + eTest.weightedRecall());
        System.out.println("f-measure: " + eTest.weightedFMeasure());
    }


//////////////// helper code /////////////////////////

    private static final double LUMINANCE_RED = 0.299D;
    private static final double LUMINANCE_GREEN = 0.587D;
    private static final double LUMINANCE_BLUE = 0.114;
    private static final int HIST_WIDTH = 256;
    private static final int HIST_HEIGHT = 100;

    /**
     * Parses pixels out of an image file, converts the RGB values to
     * its equivalent grayscale value (0-255), then constructs a
     * histogram of the percentage of counts of grayscale values.
     *
     * @param infile - the image file.
     * @return - a histogram of grayscale percentage counts.
     */
    protected static double[] buildHistogram(File infile) throws Exception {
        BufferedImage input = ImageIO.read(infile);
        int width = input.getWidth();
        int height = input.getHeight();
        List<Integer> graylevels = new ArrayList<Integer>();
        double maxWidth = 0.0D;
        double maxHeight = 0.0D;
        for (int row = 0; row < width; row++) {
            for (int col = 0; col < height; col++) {
                Color c = new Color(input.getRGB(row, col));
                int graylevel = (int) (LUMINANCE_RED * c.getRed() +
                        LUMINANCE_GREEN * c.getGreen() +
                        LUMINANCE_BLUE * c.getBlue());
                graylevels.add(graylevel);
                maxHeight++;
                if (graylevel > maxWidth) {
                    maxWidth = graylevel;
                }
            }
        }
        double[] histogram = new double[HIST_WIDTH];
        for (Integer graylevel : (new HashSet<Integer>(graylevels))) {
            int idx = graylevel;
            histogram[idx] +=
                    Collections.frequency(graylevels, graylevel) * HIST_HEIGHT / maxHeight;
        }
        return histogram;
    }
}
