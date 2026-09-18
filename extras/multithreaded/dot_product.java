package extras.multithreaded;

import java.io.*;
import java.util.*;

public class dot_product {

    // public static double dot(double[] a, double[] b) {
    // if (a.length != b.length) {
    // throw new IllegalArgumentException("vectors must be the same length");
    // }
    // double sum = 0;
    // for (int i = 0; i < a.length; i++) {
    // sum += a[i] * b[i];
    // }
    // return sum;
    // }

    public static void main(String[] args) throws InterruptedException {
        int n = 10_000_000;
        double[] a = new double[n];
        double[] b = new double[n];
        for (int i = 0; i < n; i++) {
            a[i] = 1;
            b[i] = 2;
        }

        int[] threadCounts = { 0, 4, 8, 16, 32 };
        for (int threadsCount : threadCounts) {
            long startTime = System.nanoTime();
            double result;

            if (threadsCount == 0) {
                // sequential baseline, no threads spawned
                double sum = 0;
                for (int i = 0; i < n; i++) {
                    sum += a[i] * b[i];
                }
                result = sum;
            } else {
                myThreadWorker[] allThreads = new myThreadWorker[threadsCount];
                for (int i = 0; i < threadsCount; i++) {
                    int start = i * n / threadsCount;
                    int end = (i + 1) * n / threadsCount;

                    allThreads[i] = new myThreadWorker(a, b, start, end);
                    allThreads[i].start();
                }

                double sum = 0;
                for (myThreadWorker worker : allThreads) {
                    worker.join(); // wait for this thread to finish
                    sum += worker.partialSum; // then it's safe to read
                }
                result = sum;
            }

            long endTime = System.nanoTime();
            System.out.println("threads=" + threadsCount + "  result=" + result
                    + "  time=" + (endTime - startTime) / 1_000_000.0 + " ms");
        }
    }
}

class myThreadWorker extends Thread {
    private final double[] a, b;
    private final int start, end;
    double partialSum;

    myThreadWorker(double[] a, double[] b, int start, int end) {
        this.a = a;
        this.b = b;
        this.start = start;
        this.end = end;
    }

    @Override
    public void run() {
        double sum = 0;
        for (int i = start; i < end; i++) {
            sum += a[i] * b[i];
        }
        partialSum = sum;
    }
}
