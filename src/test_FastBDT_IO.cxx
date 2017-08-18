/**
 * Thomas Keck 2014
 */

#include "FastBDT.h"
#include "FastBDT_IO.h"

#include <gtest/gtest.h>

#include <sstream>
#include <limits>

using namespace FastBDT;


class IOTest : public ::testing::Test {
    protected:
        virtual void SetUp() {}
        virtual void TearDown() {}

};


template<typename T>
::testing::AssertionResult CmpHelperFloatingPointEQNanSafe(const char* expected_expression, const char* actual_expression, T expected, T actual) {
  if(std::isnan(expected) and std::isnan(actual)) {
    return ::testing::AssertionSuccess();
  }
  return ::testing::internal::CmpHelperFloatingPointEQ(expected_expression, actual_expression, expected, actual);
}


#define EXPECT_FLOAT_EQ_NAN_SAFE(x, y)  EXPECT_PRED_FORMAT2(CmpHelperFloatingPointEQNanSafe<float>, x, y) 
#define EXPECT_DOUBLE_EQ_NAN_SAFE(x, y)  EXPECT_PRED_FORMAT2(CmpHelperFloatingPointEQNanSafe<double>, x, y) 


TEST_F(IOTest, IOVector) {

    std::vector<double> before = {0.0, 1.0, 2.5, 3.2, -1.4, 0.0};

    std::stringstream stream;
    stream << before;

    std::vector<double> after;
    stream >> after;

    EXPECT_EQ(before.size(), after.size());
    for(unsigned long i = 0; i < before.size() and i < after.size(); ++i)
        EXPECT_FLOAT_EQ(before[i], after[i]);

}

TEST_F(IOTest, IOUsingSpecialValuesFloat) {

    std::vector<float> before = {std::numeric_limits<float>::lowest(),
                                  std::numeric_limits<float>::denorm_min(),
                                  std::numeric_limits<float>::min(),
                                  std::numeric_limits<float>::max(),
                                  std::numeric_limits<float>::infinity(),
                                  -std::numeric_limits<float>::infinity(),
                                 std::numeric_limits<float>::quiet_NaN(),
                                 std::numeric_limits<float>::signaling_NaN(),
                                 0.0};

    std::stringstream stream;
    stream << before;

    std::vector<float> after;
    stream >> after;

    EXPECT_EQ(before.size(), after.size());
    for(unsigned long i = 0; i < before.size() and i < after.size(); ++i)
        EXPECT_FLOAT_EQ_NAN_SAFE(before[i], after[i]);

}

TEST_F(IOTest, IOUsingSpecialValuesDouble) {

    std::vector<double> before = {std::numeric_limits<double>::lowest(),
                                  std::numeric_limits<double>::denorm_min(),
                                  std::numeric_limits<double>::min(),
                                  std::numeric_limits<double>::max(),
                                  std::numeric_limits<double>::infinity(),
                                  -std::numeric_limits<double>::infinity(),
                                 std::numeric_limits<double>::quiet_NaN(),
                                 std::numeric_limits<double>::signaling_NaN(),
                                 0.0};

    std::stringstream stream;
    stream << before;

    std::vector<double> after;
    stream >> after;

    EXPECT_EQ(before.size(), after.size());
    for(unsigned long i = 0; i < before.size() and i < after.size(); ++i)
        EXPECT_DOUBLE_EQ_NAN_SAFE(before[i], after[i]);

}

TEST_F(IOTest, IOFeatureBinning) {

    std::vector<double> binning = { 1.0f, 7.0f, 4.0f, 10.0f, 12.0f }; 
    FeatureBinning<double> before(2, binning);
    const auto &before_binning = before.GetBinning();

    std::stringstream stream;
    stream << before;

    auto after = readFeatureBinningFromStream<double>(stream);
    const auto &after_binning = after.GetBinning();

    EXPECT_EQ(before.GetNLevels(), after.GetNLevels());
    EXPECT_EQ(before_binning.size(), after_binning.size());
    for(unsigned long i = 0; i < before_binning.size() and i < after_binning.size(); ++i)
        EXPECT_FLOAT_EQ_NAN_SAFE(before_binning[i], after_binning[i]);

}

TEST_F(IOTest, IOFeatureBinningVector) {

    std::vector<double> binning1 = { 1.0f, 7.0f, 4.0f, 10.0f, 12.0f }; 
    std::vector<double> binning2 = { 6.0f, 7.0f, 2.0f, 12.0f, 12.0f }; 
    std::vector<FeatureBinning<double>> before = {FeatureBinning<double>(2, binning1),
                                                  FeatureBinning<double>(2, binning2)};

    std::stringstream stream;
    stream << before;

    std::vector<FeatureBinning<double>> after;
    stream >> after;

    EXPECT_EQ(before.size(), after.size());
    for(unsigned long j = 0; j < before.size() and j < after.size(); ++j) {

        auto &before_featureBinning = before[j];
        auto &after_featureBinning = after[j];
        const auto &after_binning = after_featureBinning.GetBinning();
        const auto &before_binning = before_featureBinning.GetBinning();

        EXPECT_EQ(before_featureBinning.GetNLevels(), after_featureBinning.GetNLevels());
        EXPECT_EQ(before_binning.size(), after_binning.size());
        for(unsigned long i = 0; i < before_binning.size() and i < after_binning.size(); ++i)
            EXPECT_FLOAT_EQ_NAN_SAFE(before_binning[i], after_binning[i]);

    }

}

TEST_F(IOTest, IOCut) {

    Cut<unsigned long> before;
    before.feature = 1;
    before.gain = 3.4;
    before.index = 5;
    before.valid = true;

    std::stringstream stream;
    stream << before;

    Cut<unsigned long> after;
    stream >> after;

    EXPECT_EQ(before.feature, after.feature);
    EXPECT_EQ(before.gain, after.gain);
    EXPECT_EQ(before.index, after.index);
    EXPECT_EQ(before.valid, after.valid);

}

TEST_F(IOTest, IOCutSpecialValuesFloat) {
    
    std::vector<float> values = {std::numeric_limits<float>::lowest(),
                                 std::numeric_limits<float>::denorm_min(),
                                 std::numeric_limits<float>::min(),
                                 std::numeric_limits<float>::max(),
                                 std::numeric_limits<float>::infinity(),
                                 -std::numeric_limits<float>::infinity(),
                                 std::numeric_limits<float>::quiet_NaN(),
                                 std::numeric_limits<float>::signaling_NaN(),
                                 0.0};

    for(auto &f : values) {
      Cut<float> before;
      before.feature = 1;
      before.gain = 3.4;
      before.index = f;
      before.valid = true;

      std::stringstream stream;
      stream << before;

      Cut<float> after;
      stream >> after;

      EXPECT_EQ(before.feature, after.feature);
      EXPECT_FLOAT_EQ(before.gain, after.gain);
      EXPECT_FLOAT_EQ_NAN_SAFE(before.index, after.index);
      EXPECT_EQ(before.valid, after.valid);
    }

}

TEST_F(IOTest, IOCutSpecialValuesDouble) {
    
    std::vector<double> values = {std::numeric_limits<double>::lowest(),
                                 std::numeric_limits<double>::denorm_min(),
                                 std::numeric_limits<double>::min(),
                                 std::numeric_limits<double>::max(),
                                 std::numeric_limits<double>::infinity(),
                                 -std::numeric_limits<double>::infinity(),
                                 std::numeric_limits<double>::quiet_NaN(),
                                 std::numeric_limits<double>::signaling_NaN(),
                                 0.0};

    for(auto &f : values) {
      Cut<double> before;
      before.feature = 1;
      before.gain = 3.4;
      before.index = f;
      before.valid = true;

      std::stringstream stream;
      stream << before;

      Cut<double> after;
      stream >> after;

      EXPECT_EQ(before.feature, after.feature);
      EXPECT_FLOAT_EQ(before.gain, after.gain);
      EXPECT_DOUBLE_EQ_NAN_SAFE(before.index, after.index);
      EXPECT_EQ(before.valid, after.valid);
    }

}
            
TEST_F(IOTest, IOTree) {

    Cut<unsigned long> cut1, cut2, cut3;
    cut1.feature = 0;
    cut1.index = 5;
    cut1.valid = true;
    cut1.gain = -3.0;
    cut2.feature = 1;
    cut2.index = 9;
    cut2.gain = 1.0;
    cut2.valid = true;
    cut3.feature = 0;
    cut3.index = 1;
    cut3.gain = 0.0;
    cut3.valid = false;
    
    std::vector<Cut<unsigned long>> before_cuts = {cut1, cut2, cut3};
    std::vector<Weight> before_nEntries = { 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0 };
    std::vector<Weight> before_purities = { 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7 };
    std::vector<Weight> before_boostWeights = { 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0};
    Tree<unsigned long> before(before_cuts, before_nEntries, before_purities, before_boostWeights);            
    
    std::stringstream stream;
    stream << before;

    auto after = readTreeFromStream<unsigned long>(stream);
    const auto &after_cuts = after.GetCuts();
    const auto &after_purities = after.GetPurities();
    const auto &after_boostWeights = after.GetBoostWeights();
    const auto &after_nEntries = after.GetNEntries();

    EXPECT_EQ(before_cuts.size(), after_cuts.size());
    for(unsigned long i = 0; i < before_cuts.size() and i < after_cuts.size(); ++i) {
        EXPECT_FLOAT_EQ(before_cuts[i].feature, after_cuts[i].feature);
        EXPECT_FLOAT_EQ(before_cuts[i].valid, after_cuts[i].valid);
        EXPECT_FLOAT_EQ(before_cuts[i].index, after_cuts[i].index);
        EXPECT_FLOAT_EQ(before_cuts[i].gain, after_cuts[i].gain);
    }
    
    EXPECT_EQ(before_purities.size(), after_purities.size());
    for(unsigned long i = 0; i < before_purities.size() and i < after_purities.size(); ++i)
        EXPECT_FLOAT_EQ(before_purities[i], after_purities[i]);
    
    EXPECT_EQ(before_boostWeights.size(), after_boostWeights.size());
    for(unsigned long i = 0; i < before_boostWeights.size() and i < after_boostWeights.size(); ++i)
        EXPECT_FLOAT_EQ(before_boostWeights[i], after_boostWeights[i]);
    
    EXPECT_EQ(before_nEntries.size(), after_nEntries.size());
    for(unsigned long i = 0; i < before_nEntries.size() and i < after_nEntries.size(); ++i)
        EXPECT_FLOAT_EQ(before_nEntries[i], after_nEntries[i]);

}

TEST_F(IOTest, IOForest) {

    Cut<unsigned long> cut1, cut2, cut3, cut4;
    cut1.feature = 0;
    cut1.index = 5;
    cut1.valid = true;
    cut1.gain = -3.0;
    cut2.feature = 1;
    cut2.index = 9;
    cut2.gain = 1.0;
    cut2.valid = true;
    cut3.feature = 0;
    cut3.index = 1;
    cut3.gain = 0.0;
    cut3.valid = false;
    cut4.feature = 2;
    cut4.index = 3;
    cut4.valid = true;
    cut4.gain = 1.61;
    
    std::vector<Weight> nEntries = { 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0 };
    
    Forest<unsigned long> before(0.5, 1.6, true);
    before.AddTree(Tree<unsigned long>({cut1, cut2, cut3}, nEntries, { 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7 }, { 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0}));
    before.AddTree(Tree<unsigned long>({cut1, cut4, cut3}, nEntries, { 0.6, 0.2, 0.5, 0.4, 0.5, 0.6, 0.7 }, { 2.0, 2.0, 3.0, 5.0, 5.0, 6.0, 1.0}));
    const auto &before_forest = before.GetForest();

    std::stringstream stream;
    stream << before;

    auto after = readForestFromStream<unsigned long>(stream);
    const auto &after_forest = after.GetForest();

    EXPECT_EQ(before.GetTransform2Probability(), after.GetTransform2Probability());
    EXPECT_EQ(before.GetF0(), after.GetF0());
    EXPECT_EQ(before.GetShrinkage(), after.GetShrinkage());

    EXPECT_EQ(before_forest.size(), after_forest.size());
    for(unsigned long j = 0; j < before_forest.size() and j < after_forest.size(); ++j) {

        auto &before_tree = before_forest[j];
        const auto &before_cuts = before_tree.GetCuts();
        const auto &before_purities = before_tree.GetPurities();
        const auto &before_boostWeights = before_tree.GetBoostWeights();
        const auto &before_nEntries = before_tree.GetNEntries();
        
        auto &after_tree = after_forest[j];
        const auto &after_cuts = after_tree.GetCuts();
        const auto &after_purities = after_tree.GetPurities();
        const auto &after_boostWeights = after_tree.GetBoostWeights();
        const auto &after_nEntries = after_tree.GetNEntries();

        EXPECT_EQ(before_cuts.size(), after_cuts.size());
        for(unsigned long i = 0; i < before_cuts.size() and i < after_cuts.size(); ++i) {
            EXPECT_FLOAT_EQ(before_cuts[i].feature, after_cuts[i].feature);
            EXPECT_FLOAT_EQ(before_cuts[i].valid, after_cuts[i].valid);
            EXPECT_FLOAT_EQ(before_cuts[i].index, after_cuts[i].index);
            EXPECT_FLOAT_EQ(before_cuts[i].gain, after_cuts[i].gain);
        }
        
        EXPECT_EQ(before_purities.size(), after_purities.size());
        for(unsigned long i = 0; i < before_purities.size() and i < after_purities.size(); ++i)
            EXPECT_FLOAT_EQ(before_purities[i], after_purities[i]);
        
        EXPECT_EQ(before_boostWeights.size(), after_boostWeights.size());
        for(unsigned long i = 0; i < before_boostWeights.size() and i < after_boostWeights.size(); ++i)
            EXPECT_FLOAT_EQ(before_boostWeights[i], after_boostWeights[i]);
        
        EXPECT_EQ(before_nEntries.size(), after_nEntries.size());
        for(unsigned long i = 0; i < before_nEntries.size() and i < after_nEntries.size(); ++i)
            EXPECT_FLOAT_EQ(before_nEntries[i], after_nEntries[i]);
    }
}
