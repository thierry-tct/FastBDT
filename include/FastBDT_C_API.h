/**
 * Thomas Keck 2015
 */

#include "FastBDT.h"
#include "FastBDT_IO.h"
#include "Classifier.h"

extern "C" {

    void PrintVersion();

    struct Expertise {
      FastBDT::Classifier classifier;
    };
      
    void* Create();

    void SetBinning(void *ptr, unsigned long* binning, unsigned long size);
    void SetPurityTransformation(void *ptr, bool* purityTransformation, unsigned long size);
    
    void SetNTrees(void *ptr, unsigned long nTrees);
    unsigned long GetNTrees(void *ptr);
    
    void SetDepth(void *ptr, unsigned long depth);
    unsigned long GetDepth(void *ptr);
    
    void SetNumberOfFlatnessFeatures(void *ptr, unsigned long numberOfFlatnessFeatures);
    unsigned long GetNumberOfFlatnessFeatures(void *ptr);
    
    void SetSubsample(void *ptr, double subsample);
    double GetSubsample(void *ptr);
    
    void SetShrinkage(void *ptr, double shrinkage);
    double GetShrinkage(void *ptr);
    
    void SetFlatnessLoss(void *ptr, double flatnessLoss);
    double GetFlatnessLoss(void *ptr);

    void SetTransform2Probability(void *ptr, bool transform2probability);
    bool GetTransform2Probability(void *ptr);
    
    void SetSPlot(void *ptr, bool sPlot);
    bool GetSPlot(void *ptr);
    
    void Delete(void *ptr);
    
    void Fit(void *ptr, float *data_ptr, float *weight_ptr, bool *target_ptr, unsigned long nEvents, unsigned long nFeatures);

    void Load(void* ptr, char *weightfile);

    float Predict(void *ptr, float *array);

    void PredictArray(void *ptr, float *array, float *result, unsigned long nEvents);

    void Save(void* ptr, char *weightfile);
    
    struct VariableRanking {
        std::map<unsigned long, double> ranking;
    }; 

    void* GetVariableRanking(void* ptr);
    
    void* GetIndividualVariableRanking(void* ptr, float *array);
    
    unsigned long ExtractNumberOfVariablesFromVariableRanking(void* ptr);
    
    double ExtractImportanceOfVariableFromVariableRanking(void* ptr, unsigned long iFeature);
    
    void DeleteVariableRanking(void* ptr);

}
