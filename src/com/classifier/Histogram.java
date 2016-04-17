package com.classifier;

import javax.imageio.ImageIO;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.io.File;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashSet;
import java.util.List;

/**
 * Created by claudivan on 4/17/16.
 */
public class Histogram {

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
