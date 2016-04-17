package com.classifier;

import weka.core.Attribute;
import weka.core.FastVector;
import weka.core.Instance;
import weka.core.Instances;

import java.io.File;
import java.util.List;

/**
 * Created by marcos on 4/17/16.
 */
public class ClassifierSetBuilder {
    private static final int CAPACITY = 257;
    private static final int INDEX = 256;
    private Instances set;
    private FastVector wekaAttributes;

    public ClassifierSetBuilder(FastVector classes) {
        FastVector wekaAttributes = new FastVector(CAPACITY);
        for (int i = 0; i < INDEX; i++) {
            Attribute attr = new Attribute("numeric" + i);
            wekaAttributes.addElement(attr);
        }
        Attribute attr = new Attribute("classes", classes);

        wekaAttributes.addElement(attr);
        Instances trainingSet = new Instances("Rel", wekaAttributes, 1);
        trainingSet.setClassIndex(INDEX);
        this.set = trainingSet;
        this.wekaAttributes = wekaAttributes;
    }

    public void buildSet(String folderName, String clazz, List<String> files) throws Exception {
        File folder = new File(folderName);
        File[] listOfFiles = folder.listFiles();
        for (File f : listOfFiles) {
            if (files != null) {
                files.add(f.getPath());
            }
            double[] histogram = Histogram.buildHistogram(f);
            createSet(wekaAttributes, histogram, clazz);
        }
    }

    private void createSet(FastVector wekaAttributes, double[] histogram, String classe) {

        Instance imageInstance = new Instance(CAPACITY);
        for (int i = 0; i < histogram.length; i++) {
            imageInstance.setValue((Attribute) wekaAttributes.elementAt(i), histogram[i]);
        }
        if (!classe.isEmpty()) {
            imageInstance.setValue((Attribute) wekaAttributes.elementAt(INDEX), classe);
        }
        this.set.add(imageInstance);
    }

    public Instances getSet() {
        return this.set;
    }
}
