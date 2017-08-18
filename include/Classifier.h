/*
 * Thomas Keck 2017
 *
 * Simplified sklearn interface
 */

#pragma once

#include "FastBDT.h"
#include "FastBDT_IO.h"

#include <vector>

namespace FastBDT {
class Classifier {

  public:
      /*
       * Explicitly activate default/copy constructor and assign operator.
       * This was a request of a user.
       */
      Classifier() = default;
      Classifier(const Classifier&) = default;
      Classifier& operator=(const Classifier &) = default;

      Classifier(std::istream& stream) {

        stream >> m_version;
        stream >> m_nTrees;
        stream >> m_depth;
        stream >> m_binning;
        stream >> m_shrinkage;
        stream >> m_subsample;
        stream >> m_sPlot;
        stream >> m_flatnessLoss;
        stream >> m_purityTransformation;
        stream >> m_transform2probability;
        stream >> m_featureBinning;
        stream >> m_purityBinning;
        stream >> m_numberOfFeatures;
        stream >> m_numberOfFinalFeatures;
        stream >> m_numberOfFlatnessFeatures;
        stream >> m_can_use_fast_forest;
        m_fast_forest = readForestFromStream<float>(stream);
        m_binned_forest = readForestFromStream<unsigned long>(stream);

      }

      friend std::ostream& operator<<(std::ostream& stream, const Classifier& classifier);

			Classifier(unsigned long nTrees, unsigned long depth, std::vector<unsigned long> binning, double shrinkage = 0.1, double subsample = 1.0, bool sPlot = false, double flatnessLoss = -1.0, std::vector<bool> purityTransformation = {}, unsigned long numberOfFlatnessFeatures=0, bool transform2probability=true) :
        m_nTrees(nTrees), m_depth(depth), m_binning(binning), m_shrinkage(shrinkage), m_subsample(subsample), m_sPlot(sPlot), m_flatnessLoss(flatnessLoss), m_purityTransformation(purityTransformation), m_numberOfFlatnessFeatures(numberOfFlatnessFeatures), m_transform2probability(transform2probability), m_can_use_fast_forest(true) { }

      void Print();

      unsigned long GetNTrees() const { return m_nTrees; }
      void SetNTrees(unsigned long nTrees) { m_nTrees = nTrees; }
      
      unsigned long GetDepth() const { return m_depth; }
      void SetDepth(unsigned long depth) { m_depth = depth; }
      
      unsigned long GetNumberOfFlatnessFeatures() const { return m_numberOfFlatnessFeatures; }
      void SetNumberOfFlatnessFeatures(unsigned long numberOfFlatnessFeatures) { m_numberOfFlatnessFeatures = numberOfFlatnessFeatures; }

      unsigned long GetNFeatures() const { return m_numberOfFeatures; }

      double GetShrinkage() const { return m_shrinkage; }
      void SetShrinkage(double shrinkage) { m_shrinkage = shrinkage; }
      
      double GetSubsample() const { return m_subsample; }
      void SetSubsample(double subsample) { m_subsample = subsample; }
      
      bool GetSPlot() const { return m_sPlot; }
      void SetSPlot(bool sPlot) { m_sPlot = sPlot; }
      
      bool GetTransform2Probability() const { return m_transform2probability; }
      void SetTransform2Probability(bool transform2probability) { m_transform2probability = transform2probability; }
      
      std::vector<unsigned long> GetBinning() const { return m_binning; }
      void SetBinning(std::vector<unsigned long> binning) { m_binning = binning; }

      std::vector<bool> GetPurityTransformation() const { return m_purityTransformation; }
      void SetPurityTransformation(std::vector<bool> purityTransformation) { m_purityTransformation = purityTransformation; }

      double GetFlatnessLoss() const { return m_flatnessLoss; }
      void SetFlatnessLoss(double flatnessLoss) { m_flatnessLoss = flatnessLoss; }
			
      void fit(const std::vector<std::vector<float>> &X, const std::vector<bool> &y, const std::vector<Weight> &w);

      float predict(const std::vector<float> &X) const;
      
      std::map<unsigned long, double> GetVariableRanking() const;
      
      std::map<unsigned long, double> GetIndividualVariableRanking(const std::vector<float> &X) const;

      std::map<unsigned long, unsigned long> GetFeatureMapping() const;
  
      std::map<unsigned long, double> MapRankingToOriginalFeatures(std::map<unsigned long, double> ranking) const;

  private:
    unsigned long m_version = 1;
    unsigned long m_nTrees = 100;
    unsigned long m_depth = 3;
    std::vector<unsigned long> m_binning;
    double m_shrinkage = 0.1;
    double m_subsample = 0.5;
    bool m_sPlot = true;
    double m_flatnessLoss = -1;
    std::vector<bool> m_purityTransformation;
    unsigned long m_numberOfFlatnessFeatures = 0;
    bool m_transform2probability = true;
    unsigned long m_numberOfFeatures = 0;
    unsigned long m_numberOfFinalFeatures = 0;
    std::vector<FeatureBinning<float>> m_featureBinning;
    std::vector<PurityTransformation> m_purityBinning;

    bool m_can_use_fast_forest = true;
    Forest<float> m_fast_forest;
    Forest<unsigned long> m_binned_forest;

};

std::ostream& operator<<(std::ostream& stream, const Classifier& classifier);

}
