package com.pedropathing.pathgen;


import com.pedropathing.follower.FollowerConstants;
import com.pedropathing.localization.Pose;
import com.pedropathing.pathgen.BezierCurve;
import com.pedropathing.pathgen.MathFunctions;
import org.apache.commons.math3.analysis.MultivariateFunction;
import org.apache.commons.math3.optim.InitialGuess;
import org.apache.commons.math3.optim.MaxEval;
import org.apache.commons.math3.optim.PointValuePair;
import org.apache.commons.math3.optim.SimpleValueChecker;
import org.apache.commons.math3.optim.nonlinear.scalar.GoalType;
import org.apache.commons.math3.optim.nonlinear.scalar.ObjectiveFunction;
import org.apache.commons.math3.optim.nonlinear.scalar.MultivariateFunctionMappingAdapter;
import org.apache.commons.math3.optim.nonlinear.scalar.MultivariateFunctionPenaltyAdapter;
import org.apache.commons.math3.optim.nonlinear.scalar.noderiv.NelderMeadSimplex;
import org.apache.commons.math3.optim.nonlinear.scalar.noderiv.SimplexOptimizer;

import java.awt.Point;
import java.awt.geom.Point2D;
import java.io.File;
import java.lang.reflect.Array;
import java.util.ArrayList;
import java.util.List;



public class OptimisedBezierCurve extends BezierCurve{

    final List<Point> obstaclePoints;
    private final double penaltyMultiplier;
    private final double robotMass;
    final File file = new File(Environment.getExternalStorageDirectory(), "FIRST/trajectories/left_start.json");


    

    /**
     * This creates an empty OptimisedBezierCurve.
     * IMPORTANT NOTE: Only use this for the constructors of classes extending this.
     */
    public OptimisedBezierCurve(double penaltyMultiplier) {
        super();
        this.penaltyMultiplier = penaltyMultiplier;
    }
    

    /**
     * This creates a new OptimisedBezierCurve with an ArrayList of control points and generates the curve.
     *
     * @param controlPoints This is the ArrayList of control points that define the BezierCurve.
     */
    public OptimisedBezierCurve(ArrayList<Point> controlPoints, double penaltyMultiplier, List<Point> obstaclePoints, double robotMass) {
        super(controlPoints);
        this.penaltyMultiplier = penaltyMultiplier;
        this.obstaclePoints = obstaclePoints;
        this.robotMass = robotMass;
        // Optimise the control points
        double[] optimizedPoints = optimiseBezierCurve(controlPointsToDoubleArray(controlPoints));
        // Update control points with optimised ones
        if (optimizedPoints != null) {
            List<Point> newPoints = arrayToControlPoints(optimizedPoints);
            // Keep first and last points the same
            newPoints.set(0, controlPoints.get(0));
            newPoints.set(newPoints.size() - 1, controlPoints.get(controlPoints.size() - 1));
            // Reinitialise with optimised points
            controlPoints.clear();
            controlPoints.addAll(newPoints);
            initialize();
        }
    }

    /**
     * This creates a new OptimisedBezierCurve with specified control points and generates the curve.
     *
     * @param controlPoints This is the specified control points that define the BezierCurve.
     */
    public OptimisedBezierCurve(double penaltyMultiplier, List<Point> obstaclePoints, double robotMass, Point... controlPoints) {
        super(controlPoints);
        this.penaltyMultiplier = penaltyMultiplier;
        this.obstaclePoints = obstaclePoints;
        this.robotMass = robotMass;
        // Optimise the control points
        double[] optimizedPoints = optimiseBezierCurve(controlPointsToDoubleArray(controlPoints));
        // Update control points with optimized ones
        if (optimizedPoints != null) {
            List<Point> newPoints = arrayToControlPoints(optimizedPoints);
            // Keep first and last points the same
            newPoints.set(0, getFirstControlPoint());
            newPoints.set(newPoints.size() - 1, getLastControlPoint());

            // Get the internal ArrayList from the parent class and update it
            ArrayList<Point> internalControlPoints = getControlPoints();
            internalControlPoints.clear();
            internalControlPoints.addAll(newPoints);
            // Reinitialise with optimised points
            initialize();
        }
    }

    /**
     * This creates a new OptimisedBezierCurve with specified control poses and generates the curve.
     *
     * @param controlPoses This is the specified control poses that define the BezierCurve.
     */
    public OptimisedBezierCurve(double penaltyMultiplier,List<Point> obstaclePoints, double robotMass, Pose... controlPoses) {
        super(controlPoses);
        this.penaltyMultiplier = penaltyMultiplier;
        this.obstaclePoints = obstaclePoints;
        this.robotMass = robotMass;
        // Optimise the control points
        double[] optimizedPoints = optimiseBezierCurve(controlPointsToDoubleArray(controlPoses));
        // Update control points with optimised ones
        if (optimizedPoints != null) {
            List<Point> newPoints = arrayToControlPoints(optimizedPoints);
            // Keep first and last points the same
            newPoints.set(0, getFirstControlPoint());
            newPoints.set(newPoints.size() - 1, getLastControlPoint());
            // Get the internal ArrayList from the parent class and update it
            ArrayList<Point> internalControlPoints = getControlPoints();
            internalControlPoints.clear();
            internalControlPoints.addAll(newPoints);
            // Reinitialise with optimised points
            initialize();
        }
    }

    /**
     * Converts the ArrayList of Point objects to a flat double array.
     * The resulting array contains alternating x and y coordinates: [x1, y1, x2, y2, ...]
     * 
     * @return A double array containing all control points' coordinates
     */
    public double[] controlPointsToDoubleArray() {
        ArrayList<Point> points = getControlPoints();
        double[] result = new double[points.size() * 2];
        
        for (int i = 0; i < points.size(); i++) {
            Point point = points.get(i);
            result[i * 2] = point.getX();      // X coordinate
            result[i * 2 + 1] = point.getY();   // Y coordinate
        }
        
        return result;
    }
    
    /**
     * Converts an array of Point objects to a flat double array.
     * The resulting array contains alternating x and y coordinates: [x1, y1, x2, y2, ...]
     * 
     * @param points Array of Point objects
     * @return A double array containing all control points' coordinates
     */
    public double[] controlPointsToDoubleArray(Point... points) {
        double[] result = new double[points.length * 2];
        
        for (int i = 0; i < points.length; i++) {
            Point point = points[i];
            result[i * 2] = point.getX();      // X coordinate
            result[i * 2 + 1] = point.getY();   // Y coordinate
        }
        
        return result;
    }
    
    /**
     * Converts a List of Point objects to a flat double array.
     * The resulting array contains alternating x and y coordinates: [x1, y1, x2, y2, ...]
     * 
     * @param points List of Point objects
     * @return A double array containing all control points' coordinates
     */
    public double[] controlPointsToDoubleArray(List<Point> points) {
        double[] result = new double[points.size() * 2];
        
        for (int i = 0; i < points.size(); i++) {
            Point point = points.get(i);
            result[i * 2] = point.getX();      // X coordinate
            result[i * 2 + 1] = point.getY();   // Y coordinate
        }
        
        return result;
    }
    
    /**
     * Converts an array of Pose objects to a flat double array.
     * The resulting array contains alternating x and y coordinates: [x1, y1, x2, y2, ...]
     * 
     * @param poses Array of Pose objects
     * @return A double array containing all control points' coordinates
     */
    public double[] controlPointsToDoubleArray(Pose... poses) {
        double[] result = new double[poses.length * 2];
        
        for (int i = 0; i < poses.length; i++) {
            Pose pose = poses[i];
            Point point = new Point(pose);
            result[i * 2] = point.getX();      // X coordinate
            result[i * 2 + 1] = point.getY();   // Y coordinate
        }
        
        return result;
    }

    /**
     * Converts a flat double array of coordinates to a List of Point objects.
     * The input array should contain alternating x and y coordinates: [x1, y1, x2, y2, ...]
     * 
     * @param controlPoints A double array containing coordinates
     * @return A List of Point objects
     */
    public List<Point> arrayToControlPoints(double[] controlPoints) {
        List<Point> points = new ArrayList<>();
        for (int i = 0; i < controlPoints.length; i += 2) {
            points.add(new Point(controlPoints[i], controlPoints[i + 1]));
        }
        return points;
    }
    

    public double potentialFunction(double strength, double radius, Point pos) {
        double potential = 0;
        for (Point obs : obstaclePoints) {
            double dist = pos.distance(obs);
            if (dist < radius) {
                potential += strength * Math.pow((dist), 10);
            }
        }
        return potential;
    }

    /**
     * Optimises the control points of a Bezier curve to minimize a cost function
     * based on the curve's derivatives and curvature.
     * 
     * @param controlPoints Initial control points as a flat array [x1, y1, x2, y2, ...]
     * @return Optimised conrol points as a flat array
     */
    public double[] optimiseBezierCurve(double[] controlPoints) {
        final BezierCurve bezierCurve = new BezierCurve(arrayToControlPoints(controlPoints));
        final List<Point> points = arrayToControlPoints(controlPoints);


        
        MultivariateFunction costFunction = new MultivariateFunction() {
            @Override
            public double value(double[] x) {
                double action = 0;
                double N = 5000;
                double shapePenalty = 0;
                // Create a temporary BezierCurve with the current optimisation parameters
                List<Point> tempPoints = arrayToControlPoints(x);
                tempPoints.set(0, points.get(0));
                tempPoints.set(tempPoints.size() - 1, points.get(points.size() - 1));

                BezierCurve tempCurve = new BezierCurve(new ArrayList<>(tempPoints));

                // Physical constraint penalties
                double maxCurvaturePenalty = 0;
                double velocityConstraintPenalty = 0;
                double accelerationConstraintPenalty = 0;
                
                for (int i = 0; i < N; i++) {
                    double t = i / (N - 1);


                    
                    Vector D1 = tempCurve.getDerivative(t);
                    Vector D2 = tempCurve.getSecondDerivative(t);
                    double curvature = tempCurve.getCurvature(t);



                    double kineticE = 0.5*robotMass*Math.pow(D1.dot(D1), 2);

                    Point samplePoint = new Point((int) tempCurve.getPoint(t).x, (int) tempCurve.getPoint(t).y);


                    
                    //Set the raidus to the radius of the robot
                    double potentialE = potentialFunction(1000, 10, samplePoint);
                    

                    double lagrangian =  kineticE + potentialE;
                    action += lagrangian*t;
                }
                
                // Add shape penalty (to keep optimized curve close to original)
                for (int i = 1; i < tempPoints.size()-1; i++){
                    Vector delta = tempPoints.get(i).subtract(points.get(i));
                    shapePenalty += delta.dot(delta);
                }
                
                // Combine all penalties with appropriate weights
                return action + penaltyMultiplier * shapePenalty;
            }
        };
        
        try {
            //int n = controlPoints.length; // This is actually 2 * numControlPoints
            //int m = Math.min((n + 1) * (n + 2) / 2, 2 * n + 1); // Safe upper bound
            SimplexOptimizer optimizer = new SimplexOptimizer(1e-8, 1e-8);
            //BOBYQAOptimizer optimizer = new BOBYQAOptimizer(m, 1.0, 1e-6);
            NelderMeadSimplex simplex = new NelderMeadSimplex(controlPoints.length);

            PointValuePair result = optimizer.optimize(
                new MaxEval(5000),
                new ObjectiveFunction(costFunction),
                GoalType.MINIMIZE,
                new InitialGuess(controlPoints),
                simplex
            );
            
            return result.getPoint();
        } catch (Exception e) {
            e.printStackTrace();
            return null;
        }
    }
}



