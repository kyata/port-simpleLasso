#include <iostream>
#include <cstdint>
#include "pub_simpleLasso.h"

#if 0
def soft_threshold(X: np.ndarray, thresh: float):
    return np.where(np.abs(X) <= thresh, 0, X - thresh * np.sign(X))


def coordinate_descent(X: np.ndarray, y: np.ndarray, alpha: float = 1.0, n_iter: int = 1000) -> np.ndarray:
    n_samples = X.shape[0]      // X[0]の要素数を取得
    n_features = X.shape[1]     // X[1]の要素数(を取得
    w = np.zeros(n_features)    // 重みを0の値で初期化
    for _ in range(n_iter):     // loop n_iter:1000
        for j in range(n_features): // loop times n_features
            w[j] = 0.0              // 重みテーブルを初期化
            r_j = y - np.dot(X, w)  // YからX・Wの内積を引いてr_jに代入
            // Xとr_jの内積がしきい値以下なら
            w[j] = soft_threshold(np.dot(X[:, j], r_j) / n_samples, alpha)
    return w
#endif


int32_t coordinateDescent()
{
    return 1;
}

#if 0
INT32 softThreashold( float x, float y float thresh )
{

}

INT32 coordinateDescent(  )
{
    sampleNums = sizeof()
    featureNums = 
}

INT32 dotProduct(  )
{

}
#endif