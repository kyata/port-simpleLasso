#include "gtest/gtest.h"
#include "pub_simpleLasso.h"

class TestSimpleLasso : public ::testing::Test
{
public:
    const int32_t kSamples;      // サンプル数 行
    const int32_t kFeatures;     // 特徴量 列

protected:
    TestSimpleLasso() : kSamples(100), kFeatures(100) {}
    virtual void SetUp()
    {
        lhs_ = Eigen::MatrixXd::Random(kSamples, kFeatures);
        rhs_ = Eigen::MatrixXd::Random(kSamples, kFeatures);
    }

    virtual void TearDown() {}

    Eigen::MatrixXd lhs_;   // 検算用ダミー行列
    Eigen::MatrixXd rhs_;   // 検算用ダミー行列
};

TEST_F(TestSimpleLasso, getSignTest)
{
    EXPECT_EQ(getSign(0.1),  1);
    EXPECT_EQ(getSign(0), 0);
    EXPECT_EQ(getSign(-0.1), -1);
}

TEST_F(TestSimpleLasso, innerProductTest)
{
    std::vector<double> vec {
        0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0
    };

    double prod = innerProduct(vec, vec);
    EXPECT_EQ(285.0, prod);
}

TEST_F(TestSimpleLasso, getVectorTest)
{
    int32_t idx = 2;

    std::vector<double> vec1 = getVector(lhs_, idx, GET_VECTOR_TYPE_COL);
    std::vector<double> vec2 = getVector(lhs_, idx, GET_VECTOR_TYPE_ROW);

    for( auto i=0; i<vec1.size(); i++) {
        EXPECT_EQ( vec1[i], lhs_(i, idx) );
    }

    for(auto i=0; i<vec2.size(); i++) {
        EXPECT_EQ( vec2[i], lhs_(idx, i) );
    }
}

TEST_F(TestSimpleLasso, addMatrixTest)
{
    Eigen::MatrixXd C = addMatrix(lhs_, rhs_);
    Eigen::MatrixXd CC = lhs_ + rhs_;

    EXPECT_EQ(C, CC);
}

TEST_F(TestSimpleLasso, subMatrixTest)
{
    Eigen::MatrixXd C = subMatrix(lhs_, rhs_);
    Eigen::MatrixXd CC = lhs_ - rhs_;

    EXPECT_EQ(C, CC);
}

TEST_F(TestSimpleLasso, dotMatrixTest)
{
    Eigen::MatrixXd C = dotMatrix(lhs_, rhs_);
    Eigen::MatrixXd CC = lhs_ * rhs_;

    EXPECT_EQ(C, CC);
}
