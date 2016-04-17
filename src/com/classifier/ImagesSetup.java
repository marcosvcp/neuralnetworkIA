package com.classifier;

import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.functions.MultilayerPerceptron;

import weka.core.Attribute;
import weka.core.FastVector;
import weka.core.Instance;
import weka.core.Instances;

import javax.imageio.ImageIO;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.io.File;
import java.util.*;
import java.util.List;

public class ImagesSetup {

    private static final String DIGITOS = "digitos";
    private static final String LETRAS = "letras";
    private static final String DIGITOS_LETRAS = "digitos_letras";
    private static final String SEM_CARACTERES = "sem_caracteres";
    private static final int CAPACITY = 257;
    private static final int INDEX = 256;

    public static void main(String[] a) throws Exception {
        boolean verbose = Boolean.parseBoolean(a[0]);
        String pathTrain = a[1];
        String pathTest = a[2];
        FastVector wekaAttributes = new FastVector(CAPACITY);
        for (int i = 0; i < INDEX; i++) {
            Attribute attr = new Attribute("numeric" + i);
            wekaAttributes.addElement(attr);
        }
        FastVector classes = new FastVector(4);
        classes.addElement(DIGITOS);
        classes.addElement(LETRAS);
        classes.addElement(DIGITOS_LETRAS);
        classes.addElement(SEM_CARACTERES);
        Attribute attr = new Attribute("classes", classes);

        wekaAttributes.addElement(attr);
        Instances isTrainingSet = new Instances("Rel", wekaAttributes, 1);
        isTrainingSet.setClassIndex(INDEX);
        Classifier cModel = new MultilayerPerceptron();
        String folderDigits = pathTrain + "/digitos";
        String folderLetters = pathTrain + "/letras";
        String folderBoth = pathTrain + "/digitos_letras";
        String nothing = pathTrain + "/sem_caracteres";
        buildSet(wekaAttributes, isTrainingSet, folderDigits, DIGITOS);
        buildSet(wekaAttributes, isTrainingSet, folderLetters, LETRAS);
        buildSet(wekaAttributes, isTrainingSet, folderBoth, DIGITOS_LETRAS);
        buildSet(wekaAttributes, isTrainingSet, nothing, SEM_CARACTERES);
        cModel.buildClassifier(isTrainingSet);

        Evaluation eTest = new Evaluation(isTrainingSet);
        Instances testingSet = new Instances("Reltst", wekaAttributes, 1);
        testingSet.setClassIndex(INDEX);
        String folderTestLetters = pathTest + "/letras";
        String folderTestDigits = pathTest + "/digitos";
        String folderTestBoth = pathTest + "/digitos_letras";
        String nothingTest = pathTest + "/sem_caracteres";
        buildSet(wekaAttributes, testingSet, folderTestLetters, "");
        buildSet(wekaAttributes, testingSet, folderTestDigits, "");
        buildSet(wekaAttributes, testingSet, folderTestBoth, "");
        buildSet(wekaAttributes, testingSet, nothingTest, "");
        eTest.evaluateModel(cModel, testingSet);

        if (verbose) {
            System.out.println(eTest.toSummaryString(true));
            System.out.println(eTest.toClassDetailsString());
        }
        System.out.println("precision: " + eTest.weightedPrecision());
        System.out.println("recall: " + eTest.weightedRecall());
        System.out.println("f-measure: " + eTest.weightedFMeasure());
    }

    private static void buildSet(FastVector wekaAttributes, Instances isTrainingSet, String folderName, String classe) throws Exception {
        File folder = new File(folderName);
        File[] listOfFiles = folder.listFiles();
        for (File f : listOfFiles) {
            double[] histogram = buildHistogram(f);
            createTrainingSet(isTrainingSet, wekaAttributes, histogram, classe);
        }
    }

    private static void createTrainingSet(Instances isTrainingSet, FastVector wekaAttributes, double[] histogram, String classe) {

        Instance imageInstance = new Instance(CAPACITY);
        for (int i = 0; i < histogram.length; i++) {
            imageInstance.setValue((Attribute) wekaAttributes.elementAt(i), histogram[i]);
        }
        if (!classe.isEmpty()) {
            imageInstance.setValue((Attribute) wekaAttributes.elementAt(INDEX), classe);
        }
        isTrainingSet.add(imageInstance);
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
