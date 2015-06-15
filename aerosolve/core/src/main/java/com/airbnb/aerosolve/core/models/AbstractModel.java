package com.airbnb.aerosolve.core.models;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.IOException;
import java.io.Serializable;
import java.util.List;

import com.airbnb.aerosolve.core.DebugScoreRecord;
import com.airbnb.aerosolve.core.FeatureVector;
import com.airbnb.aerosolve.core.ModelHeader;
import lombok.Getter;
import lombok.Setter;
/**
 * Created by hector_yee on 8/25/14.
 * Base class for models
 */

public abstract class AbstractModel implements Model, Serializable {

  private static final long serialVersionUID = -5011350794437028492L;

  @Getter @Setter
  protected double offset = 0.0;

  @Getter @Setter
  protected double slope = 1.0;

  // Scores a single item. The transforms should already have been applied to
  // the context and item and combined item.
  abstract public float scoreItem(FeatureVector combinedItem);

  // Debug scores a single item. These are explanations for why a model
  // came up with the score.
  abstract public float debugScoreItem(FeatureVector combinedItem,
                                       StringBuilder builder);

  abstract public List<DebugScoreRecord> debugScoreComponents(FeatureVector combinedItem);

  // Loads model from a buffered stream.
  abstract protected void loadInternal(ModelHeader header, BufferedReader reader) throws IOException;

  abstract public void save(BufferedWriter writer) throws IOException;

  // returns probability: 1 / (1 + exp(-(offset + scale * score))
  public double scoreProbability(float score) {
    return 1.0 / (1.0 + Math.exp(-(offset + slope * score)));
  }

  // Helper function for FOBOS updates.
  // http://papers.nips.cc/paper/3793-efficient-learning-using-forward-backward-splitting.pdf
  public static float fobosUpdate(
      float currWeight,
      float gradient,
      float learningRate,
      float l1Reg,
      float l2Reg,
      float ssg) {
    float etaT = learningRate / (float) Math.sqrt(ssg);
    float etaTHalf = learningRate / (float) Math.sqrt(ssg + 0.5);
    // FOBOS l2 regularization
    float wt = (currWeight - gradient * etaT) / (1.0f + l2Reg * etaTHalf);
    // FOBOS l1 regularization
    float sign = 0.0f;
    if (wt > 0.0) {
      sign = 1.0f;
    } else {
      sign = -1.0f;
    }
    float step = (float) Math.max(0.0, Math.abs(wt) - l1Reg * etaTHalf);
    return sign * step;
  }
}
