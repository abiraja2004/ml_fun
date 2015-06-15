package com.airbnb.aerosolve.training

import java.util

import com.airbnb.aerosolve.core.models.BoostedStumpsModel
import com.airbnb.aerosolve.core.models.DecisionTreeModel
import com.airbnb.aerosolve.core.Example
import com.airbnb.aerosolve.core.ModelRecord
import com.airbnb.aerosolve.core.util.Util
import com.typesafe.config.Config
import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._
import org.apache.spark.rdd.RDD
import org.slf4j.{Logger, LoggerFactory}

import scala.util.Random
import scala.collection.JavaConversions._
import scala.collection.JavaConverters._

// The decision tree is meant to be a prior for the spline model / linear model
object DecisionTreeTrainer {
  private final val log: Logger = LoggerFactory.getLogger("DecisionTreeTrainer")

  def train(sc : SparkContext,
            input : RDD[Example],
            config : Config,
            key : String) : DecisionTreeModel = {
    val candidateSize : Int = config.getInt(key + ".num_candidates")
    val rankKey : String = config.getString(key + ".rank_key")
    val rankThreshold : Double = config.getDouble(key + ".rank_threshold")
    val maxDepth : Int = config.getInt(key + ".max_depth")
    val minLeafCount : Int = config.getInt(key + ".min_leaf_items")
    val numTries : Int = config.getInt(key + ".num_tries")

    val examples = LinearRankerUtils
        .makePointwiseFloat(input, config, key)
        .map(x => Util.flattenFeature(x.example(0)))
        .filter(x => x.contains(rankKey))
        .take(candidateSize)
        .toArray
    
    val stumps = new util.ArrayList[ModelRecord]()
    stumps.append(new ModelRecord)
    buildTree(stumps, examples, 0, 0, maxDepth, rankKey, rankThreshold, numTries, minLeafCount)
    
    val model = new DecisionTreeModel()
    model.setStumps(stumps)

    model
  }
  
  def buildTree(stumps : util.ArrayList[ModelRecord],
                examples :  Array[util.Map[java.lang.String, util.Map[java.lang.String, java.lang.Double]]],
                currIdx : Int,
                currDepth : Int,
                maxDepth : Int,
                rankKey : String,
                rankThreshold : Double,
                numTries : Int,
                minLeafCount : Int) : Unit = {
    if (currDepth >= maxDepth) {
      stumps(currIdx) = makeLeaf(examples, rankKey, rankThreshold)
      return
    }
    val split = getBestSplit(examples, rankKey, rankThreshold, numTries, minLeafCount)
    if (split == None) {
      stumps(currIdx) = makeLeaf(examples, rankKey, rankThreshold)
      return
    }
    // This is a split node.
    stumps(currIdx) = split.get    
    val left = stumps.size
    stumps.append(new ModelRecord())
    val right = stumps.size
    stumps.append(new ModelRecord())
    stumps(currIdx).setLeftChild(left)
    stumps(currIdx).setRightChild(right)
    
    buildTree(stumps,
              examples.filter(x => BoostedStumpsModel.getStumpResponse(stumps(currIdx), x) == false),
              left,
              currDepth + 1,
              maxDepth,
              rankKey,
              rankThreshold,
              numTries,
              minLeafCount)
    buildTree(stumps,
              examples.filter(x => BoostedStumpsModel.getStumpResponse(stumps(currIdx), x) == true),
              right,
              currDepth + 1,
              maxDepth,
              rankKey,
              rankThreshold,
              numTries,
              minLeafCount)    
  }
  
  def makeLeaf(examples :  Array[util.Map[java.lang.String, util.Map[java.lang.String, java.lang.Double]]],
               rankKey : String,
               rankThreshold : Double) = {
    var numPos = 0.0
    var numNeg = 0.0
    for (example <- examples) {
      val label = if (example.get(rankKey).asScala.head._2 <= rankThreshold) false else true
      if (label) numPos += 1.0 else numNeg += 1.0
    }
    val rec = new ModelRecord()
    val sum = numPos + numNeg
    if (sum > 0.0) {
      // Convert from percentage positive to the -1 to 1 range
      val frac = numPos / sum
      rec.setFeatureWeight(2.0 * frac - 1.0)
    } else {
      rec.setFeatureWeight(0.0)
    }
    rec
  }

  // Returns the best split if one exists.
  def getBestSplit(examples :  Array[util.Map[java.lang.String, util.Map[java.lang.String, java.lang.Double]]],
                   rankKey : String,
                   rankThreshold : Double,
                   numTries : Int,
                   minLeafCount : Int) : Option[ModelRecord] = {
    var record : Option[ModelRecord] = None
    var best : Double = 0
    val rnd = new Random()
    for (i <- 0 until numTries) {
      // Pick an example index randomly
      val idx = rnd.nextInt(examples.size)
      val ex = examples(idx)
      val candidateOpt = getCandidateSplit(ex, rankKey, rnd)
      if (candidateOpt != None) {
        var leftPos : Double = 0.0
        var rightPos : Double = 0.0
        var leftNeg : Double = 0.0
        var rightNeg : Double = 0.0
        for (example <- examples) {
          val response = BoostedStumpsModel.getStumpResponse(candidateOpt.get, example)
          val label = if (example.get(rankKey).asScala.head._2 <= rankThreshold) false else true
          if (response) {
            if (label) {
              rightPos += 1.0
            } else {
              rightNeg += 1.0
            }
          } else {
            if (label) {
              leftPos += 1.0
            } else {
              leftNeg += 1.0
            }
          }
        }
        val rightCount = rightPos + rightNeg
        val leftCount = leftPos + leftNeg
        if (rightCount >= minLeafCount && leftCount >= minLeafCount) {
          val scale = 1.0 / (leftCount * rightCount)
          // http://en.wikipedia.org/wiki/Bhattacharyya_distance
          val bhattacharyya = math.sqrt(leftPos * rightPos * scale) + math.sqrt(leftNeg * rightNeg * scale)
          // http://en.wikipedia.org/wiki/Hellinger_distance
          val hellinger = math.sqrt(1.0 - bhattacharyya)
          if (hellinger > best) {
            best = hellinger
            record = candidateOpt
          }
        }
      }
    }
    
    record
  }
  
  // Returns a candidate split sampled from an example.
  def getCandidateSplit(ex : util.Map[java.lang.String, util.Map[java.lang.String, java.lang.Double]],
                        rankKey : String,
                        rnd : Random) : Option[ModelRecord] = {
    // Flatten the features and pick one randomly.
    val features = collection.mutable.ArrayBuffer[(String, String, Double)]()
    for (family <- ex) {
      if (!family._1.equals(rankKey)) {
        for (feature <- family._2) {
          features.append((family._1, feature._1, feature._2))
        }
      }
    }
    if (features.size == 0) {
      return None
    }
    val idx = rnd.nextInt(features.size)
    val rec = new ModelRecord()
    rec.setFeatureFamily(features(idx)._1)
    rec.setFeatureName(features(idx)._2)
    rec.setThreshold(features(idx)._3)
    Some(rec)
  }

  def trainAndSaveToFile(sc : SparkContext,
                         input : RDD[Example],
                         config : Config,
                         key : String) = {
    val model = train(sc, input, config, key)
    TrainingUtils.saveModel(model, config, key + ".model_output")
  }
}
