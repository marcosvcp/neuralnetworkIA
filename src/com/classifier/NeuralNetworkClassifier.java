package com.classifier;

import weka.classifiers.Classifier;
import weka.classifiers.functions.MultilayerPerceptron;
import weka.core.Attribute;
import weka.core.FastVector;
import weka.core.Instances;

import java.util.List;

/**
 * Created by marcos on 4/17/16.
 */
public class NeuralNetworkClassifier {

    private Classifier model;
    private ClassifierSetBuilder setBuilder;


    public NeuralNetworkClassifier(FastVector classes) {

        this.model = new MultilayerPerceptron();
        this.setBuilder = new ClassifierSetBuilder(classes);
    }

    public void buildClassifier() throws Exception {
        this.model.buildClassifier(getSet());
    }

    public Classifier getClassifier() {
        return this.model;
    }


    public Instances getSet() {
        return this.setBuilder.getSet();
    }

    public void buildSet(String folderName, String classe, List<String> files) throws Exception {
        setBuilder.buildSet(folderName, classe, files);
    }
}
