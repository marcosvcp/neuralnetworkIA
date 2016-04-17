package com.classifier;

import weka.classifiers.Evaluation;
import weka.core.FastVector;
import weka.core.Instances;

import java.util.ArrayList;
import java.util.List;

public class ImagesSetup {

    private static final String DIGITOS = "digitos";
    private static final String LETRAS = "letras";
    private static final String DIGITOS_LETRAS = "digitos_letras";
    private static final String SEM_CARACTERES = "sem_caracteres";

    static List<String> paths = new ArrayList<String>();
    static NeuralNetworkClassifier NNCTrain;
    static Evaluation eTest;
    static Instances testSet;

    static boolean verbose;

    public static void main(String[] a) throws Exception {
        verbose = Boolean.parseBoolean(a[0]);
        String pathTrain = a[1];
        String pathTest = a[2];

        FastVector classes = new FastVector(4);
        classes.addElement(DIGITOS);
        classes.addElement(LETRAS);
        classes.addElement(DIGITOS_LETRAS);
        classes.addElement(SEM_CARACTERES);

        NNCTrain = new NeuralNetworkClassifier(classes);
        String folderDigits = pathTrain + "/digitos";
        String folderLetters = pathTrain + "/letras";
        String folderBoth = pathTrain + "/digitos_letras";
        String nothing = pathTrain + "/sem_caracteres";

        NNCTrain.buildSet(folderDigits, DIGITOS, null);
        NNCTrain.buildSet(folderLetters, LETRAS, null);
        NNCTrain.buildSet(folderBoth, DIGITOS_LETRAS, null);
        NNCTrain.buildSet(nothing, SEM_CARACTERES, null);
        NNCTrain.buildClassifier();

        initTest(pathTest,classes);
        printStatistics();
    }

    static private void initTest(String pathTest, FastVector classes) throws Exception {
        eTest = new Evaluation(NNCTrain.getSet());
        ClassifierSetBuilder testBuilder = new ClassifierSetBuilder(classes);
        String folderTestLetters = pathTest + "/letras";
        String folderTestDigits = pathTest + "/digitos";
        String folderTestBoth = pathTest + "/digitos_letras";
        String nothingTest = pathTest + "/sem_caracteres";

        testBuilder.buildSet(folderTestLetters, LETRAS, paths);
        testBuilder.buildSet(folderTestDigits, DIGITOS, paths);
        testBuilder.buildSet(folderTestBoth, DIGITOS_LETRAS, paths);
        testBuilder.buildSet(nothingTest, SEM_CARACTERES, paths);

        testSet = testBuilder.getSet();
        eTest.evaluateModel(NNCTrain.getClassifier(), testSet);
    }

    private static void printStatistics() throws Exception {
        for (int i = 0; i < testSet.numInstances(); i++) {
            double pred = NNCTrain.getClassifier().classifyInstance(testSet.instance(i));
            System.out.println("ID: " + paths.get(i));
            String actual = testSet.classAttribute().value((int) testSet.instance(i).classValue());
            String predicted = testSet.classAttribute().value((int) pred);
            System.out.println("actual: " + actual);
            System.out.println("predicted: " + predicted);
            System.out.println();
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

}
