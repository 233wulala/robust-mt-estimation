"""
The source for these functions is Robust Statisitics, Huber, 2009
in general, linear regression is# have observations y and predictors A
y is multiple observations/response
x are the independent variables and is unknown
and y is a linear function of x => y = Ax
y = nobs
A = nobs * nregressors
x = nregressors
"""

import numpy as np
import numpy.linalg as linalg
import scipy.stats as stats
from typing import List, Dict, Tuple

from resistics.common.checks import parseKeywords
import warnings

def sestimateModel(A: np.ndarray, y: np.ndarray, **kwargs) -> Tuple:
    """S-estimate robust regression with leverage adjustment

    S估计通过最小化残差的M-scale来估计参数。
    M-scale通过方程 (1/n)Σρ(r_i/σ) = δ 定义。

    修改说明：
    1. 使用solve_m_scale_*函数求解M-scale（见文件末尾）
    2. 检查尺度收敛而非参数收敛
    3. 其他函数完全使用文件中已有的函数

    Parameters
    ----------
    A : np.ndarray
        Predictors, size nobs x nregressors
    y : np.ndarray
        Observations, size nobs
    intercept : bool, optional
        Add intercept term
    weights : str, optional
        Weight function type ('bisquare', 'huber', 'hampel', etc.)
    nstarts : int, optional
        Number of random starting points

    Returns
    -------
    params : np.ndarray
        Estimated parameters
    resids : np.ndarray
        Residuals = y - Ax
    scale : float
        Final robust scale estimate
    weights : np.ndarray
        Final robust weights
    """
    options = parseKeywords(defaultDictionaryS(), kwargs, printkw=False)

    n, p = A.shape
    intercept = options["intercept"]

    if intercept:
        A = np.hstack((np.ones((n, 1), dtype="complex"), A))
        p += 1

    # Leverage weights based on projection matrix
    q, r = linalg.qr(A)
    Pdiag = np.array([np.sum(q[i, :] * np.conjugate(q[i, :])).real for i in range(n)])
    Pdiag /= np.max(Pdiag)
    leverageScale = sampleMAD0(Pdiag)
    leverageWeights = getRobustLocationWeights(Pdiag / leverageScale, "bisquare")

    best_scale = np.inf
    best_params = None
    best_resids = None
    best_weights = None

    for start in range(options["nstarts"]):
        if start == 0:
            params, resids, _, _, _ = olsModel(A, y, intercept=False)
        else:
            indices = np.random.choice(n, size=min(p + 2, n), replace=False)
            A_sub = A[indices]
            y_sub = y[indices]
            if A_sub.shape[0] >= A_sub.shape[1]:
                params, _, _, _ = linalg.lstsq(A_sub, y_sub, rcond=None)
                resids = y - np.dot(A, params)
            else:
                continue

        prev_scale = np.inf

        for iteration in range(options["maxiter"]):
            # 🔥 修改1: 使用M-scale求解函数（见文件末尾）
            if options["weights"] == "huber":
                scale = solve_m_scale_huber(resids)
            else:  # bisquare或其他，默认bisquare
                scale = solve_m_scale_bisquare(resids)

            if scale < eps():
                warnings.warn("Scale too small; skipping this start.")
                break

            # 🔥 修改2: 检查尺度收敛（S估计的目标是最小化尺度）
            if iteration > 0 and scale >= prev_scale - 1e-6 * prev_scale:
                break

            prev_scale = scale

            # 计算权重（使用已有函数）
            std_resids = resids / scale
            weights = getRobustLocationWeights(std_resids, options["weights"]) * leverageWeights

            # 加权最小二乘（使用已有函数）
            A_weighted, y_weighted = weightLS(A, y, weights)
            params_new, _, _, _ = linalg.lstsq(A_weighted, y_weighted, rcond=None)

            if np.any(np.isnan(params_new)) or np.any(np.isinf(params_new)):
                break

            resids_new = y - np.dot(A, params_new)
            params = params_new
            resids = resids_new

        # 🔥 修改3: 最终尺度用M-scale
        if options["weights"] == "huber":
            final_scale = solve_m_scale_huber(resids)
        else:
            final_scale = solve_m_scale_bisquare(resids)

        if final_scale < best_scale:
            best_scale = final_scale
            best_params = params.copy()
            best_resids = resids.copy()
            best_weights = getRobustLocationWeights(resids / final_scale, options["weights"]) * leverageWeights

    if best_params is None:
        best_params, best_resids, _, _, _ = olsModel(A, y, intercept=False)
        best_scale = sampleMAD0(best_resids)
        best_weights = np.ones(n)

    return best_params, best_resids, best_scale, best_weights


def defaultDictionaryS() -> Dict:
    """S-estimate regression defaults

    Returns
    -------
    Dict
        Default S-estimation options
    """
    outDict = {}
    # outDict["weights"] = "huber"  # use existing weight functions
    outDict["weights"] = "bisquare"  # use existing weight functions
    outDict["maxiter"] = 100  # fewer iterations for S-estimate
    outDict["intercept"] = False
    outDict["nstarts"] = 10  # reasonable number of starts
    # outDict["initial"] = False  # 新增
    return outDict


def olsModel(A, y, **kwargs) -> Tuple:
    r"""Ordinary least squares

    Solves for :math:`x` where,

    .. math::
        y = Ax .

    Parameters
    ----------
    A : np.ndarray
        Predictors, size nobs*nregressors
    y : np.ndarray
        Observations, size nobs
    intercept : bool, optional
        True or False for adding an intercept term

    Returns
    -------
    params : np.ndarray
        Least squares solution
    resids : np.ndarray
        Residuals
    squareResid : np.ndarray
        Square residuals
    rank : int
        Rank of matrix A
    s : np.ndarray
        Singular values of A
    """
    options = parseKeywords(defaultDictionary(), kwargs, printkw=False)
    if options["intercept"]:
        # add a constant term for the intercept
        A = np.hstack((np.ones(shape=(A.shape[0], 1), dtype="complex"), A))
    params, squareResid, rank, s = linalg.lstsq(A, y, rcond=None)
    resids = y - np.dot(A, params)
    return params, resids, squareResid, rank, s


def calculateDistCMH(n, x, mean, covariance):
    inv = np.linalg.inv(covariance)
    dist = np.empty(shape=(n), dtype="float")
    for i in range(0, n):
        tmp = x[i, :] - mean
        dist[i] = np.sqrt(np.dot(tmp, np.dot(inv, tmp)))
    return dist


def weightLS(A: np.ndarray, y: np.ndarray, weights: np.ndarray) -> Tuple[np.ndarray]:
    r"""Transform A and y using the weights to perform a weighted least squares

    .. math::
        \sqrt{weights} y = \sqrt{weights} A x ,

    is equivalent to,

    .. math::
        A^H weights y = A^H weights A x ,

    where :math:`A^H` is the hermitian transpose.

    In this method, both y and A are multipled by the square root of the weights and then returned.

    Parameters
    ----------
    y : np.ndarray
        Observations
    A : np.ndarray
        Regressors

    Returns
    ----------
    y : np.ndarray
        Observations multipled by the square root of the weights
    A : np.ndarray
        Regressors multipled by the square root of the weights
    """
    ynew = np.sqrt(weights) * y
    Anew = np.empty(shape=A.shape, dtype="complex")
    for col in range(0, A.shape[1]):
        Anew[:, col] = np.sqrt(weights) * A[:, col]
    return Anew, ynew


def hermitianTranspose(mat: np.ndarray) -> np.ndarray:
    """Hermitian transpose (transpose and complex conjugation)

    Parameters
    ----------
    np.ndarray
        Vector, matrix to Hermitian transpose

    Returns
    -------
    np.ndarray
        Hermitian transpose
    """
    return np.conjugate(np.transpose(mat))


def initialFromDict(initDict: Dict) -> Tuple:
    """Returns initial model from provided initial model dictionary

    Helps for two stage robust regression.

    Parameters
    ----------
    Dict
        Initial model to use for robust regression with the parameters, residuals and scale estimate

    Returns
    -------
    parameters : np.ndarray

    resids : np.ndarray
        The residuals
    scale : float
        Initial estimate of scale
    """
    return initDict["params"], initDict["resids"], initDict["scale"]


def defaultDictionary() -> Dict:
    """Robust regression defaults

    Returns
    -------
    Dict
        Default regression options
    """
    outDict = {}
    outDict["weights"] = "bisquare"
    outDict["maxiter"] = maxIter()
    outDict["initial"] = False
    outDict["scale"] = False
    outDict["intercept"] = False
    return outDict


def getRobustLocationWeights(r: np.ndarray, weight: str) -> np.ndarray:
    """Robust weighting schemes

    Parameters
    ----------
    r : np.ndarray
        Residuals
    weight : str
        The type of weighting to use

    Returns
    -------
    weights : np.ndarray
        The robust weights
    """
    # the second argument, k, is a tuning constant
    if weight == "huber":
        k = 1.345
        # k = 0.5
        return huberLocationWeights(r, k)
    elif weight == "hampel":
        k = 8
        return hampelLocationWeights(r, k)
    elif weight == "trimmedMean":
        k = 2
        return trimmedMeanLocationWeights(r, k)
    elif weight == "andrewsWave":
        k = 1.339
        return andrewsWaveLocationWeights(r, k)
    elif weight == "leastsq":
        return leastSquaresLocationWeights(r)
    else:
        # use bisquare weights
        # k = 4.685
        # k = 1.0
        k = 1.547  # 根据文献
        return bisquareLocationWeights(r, k)


def huberLocationWeights(r: np.ndarray, k: float) -> np.ndarray:
    """Huber location weights

    Parameters
    ----------
    r : np.ndarray
        Residuals
    k : float
        Tuning parameter

    Returns
    -------
    weights : np.ndarray
        The robust weights
    """
    weights = np.ones(shape=r.size, dtype="complex")
    for idx, val in enumerate(np.absolute(r)):
        if val > k:
            # relying on numpy doing the right thing when dividing by zero
            weights[idx] = k / val
    return weights.real


def bisquareLocationWeights(r: np.ndarray, k: float) -> np.ndarray:
    """Bisquare location weights

    Parameters
    ----------
    r : np.ndarray
        Residuals
    k : float
        Tuning parameter

    Returns
    -------
    weights : np.ndarray
        The robust weights
    """
    ones = np.ones(shape=(r.size), dtype="complex")
    threshR = np.minimum(ones, np.absolute(r / k))
    # threshR = np.maximum(-1*ones, threshR)
    return np.power((1 - np.power(threshR, 2)), 2).real


def hampelLocationWeights(r: np.ndarray, k: float) -> np.ndarray:
    """Hampel location weights

    Parameters
    ----------
    r : np.ndarray
        Residuals
    k : float
        Tuning parameter

    Returns
    -------
    weights : np.ndarray
        The robust weights
    """
    a = k / 4
    b = k / 2
    weights = np.ones(shape=r.size, dtype="complex")
    for idx, val in enumerate(np.absolute(r)):
        if val > a and val <= b:
            weights[idx] = a / val
        if val > b and val <= k:
            weights[idx] = a * (k - val) / (val * (k - b))
        if val > k:
            weights[idx] = 0
    return weights.real


def trimmedMeanLocationWeights(r: np.ndarray, k: float) -> np.ndarray:
    """Trimmed mean location weights

    Parameters
    ----------
    r : np.ndarray
        Residuals
    k : float
        Tuning parameter

    Returns
    -------
    weights : np.ndarray
        The robust weights
    """
    weights = np.zeros(shape=r.size, dtype="complex")
    indices = np.where(np.absolute(r) <= k)
    weights[indices] = 1
    return weights.real


def andrewsWaveLocationWeights(r: np.ndarray, k: float) -> np.ndarray:
    """Andrews Wave location weights

    Parameters
    ----------
    r : np.ndarray
        Residuals
    k : float
        Tuning parameter

    Returns
    -------
    weights : np.ndarray
        The robust weights
    """
    weights = np.zeros(shape=r.size, dtype="complex")
    testVal = k * np.pi
    for idx, val in enumerate(np.absolute(r)):
        if val < testVal:
            weights[idx] = np.sin(val / k) / (val / k)
    return weights.real


def leastSquaresLocationWeights(r: np.ndarray):
    """Least squares weights, which are all equal to 1

    Parameters
    ----------
    r : np.ndarray
        Residuals

    Returns
    -------
    weights : np.ndarray
        The robust weights
    """
    return np.ones(shape=(r.size), dtype="complex")


def sampleMedian(data):
    """Calculate the median of an array

    Mean is not a robust estimator of locations as it can be broken by a single outlying value. The median is a more robust choice.

    Parameters
    ----------
    np.ndarray
        Data for which to calculate median

    Returns
    -------
    float
        The median
    """
    return np.median(data)


def sampleMAD(data):
    """Median absolute deviation

    The standard deviation is not robust against outliers, hence use the MAD.

    Parameters
    ----------
    np.ndarray
        Data for which to calculate MAD

    Returns
    -------
    float
        The MAD
    """
    absData = np.absolute(data)
    mad = sampleMedian(np.absolute(absData - sampleMedian(absData)))
    return mad / 0.67448975019608171


def sampleMAD0(data):
    """Median absolute deviation using an estimate of the location as 0

    When the location estimate is zero (rather than the median), the MAD essentially reduces to a median. This should be over non zero data. Useful for calculating variance of residuals.

    Parameters
    ----------
    np.ndarray
        Data for which to calculate MAD. This is often residuals when using 0 as an estimate of location.

    Returns
    -------
    float
        The MAD using zero as an esimate of location
    """
    absData = np.absolute(data)  # np.absolute逐个计算元素的绝对值
    inputIndices = np.where(absData != 0.0)
    mad = sampleMedian(absData[inputIndices])
    # mad = sampleMedian(np.absolute(data))
    return mad / 0.67448975019608171


def eps() -> float:
    """Small number

    Returns
    -------
    float
        A small number for quitting robust regression
    """
    return 0.0001


def maxIter() -> int:
    """Maximum number of iterations

    Returns
    -------
    int
        The maximum number of iterations
    """
    return 100


# ===============================
# S估计所需的辅助函数（新增）
# ===============================

def get_rho_bisquare(u: np.ndarray, c: float = 4.685) -> np.ndarray:
    """Tukey bisquare ρ函数

    S估计的M-scale方程需要ρ(u)函数，这是损失函数。
    与权重函数w(u)的关系: w(u) = ρ'(u)/u

    bisquare的w(u) = (1-(u/c)^2)^2 for |u|<=c, 0 otherwise
    对应的ρ(u) = (c^2/6)[1-(1-(u/c)^2)^3] for |u|<=c, c^2/6 otherwise

    Parameters
    ----------
    u : np.ndarray
        标准化残差
    c : float
        调节参数，默认4.685（95%效率）

    Returns
    -------
    np.ndarray
        ρ(u) 值
    """
    rho = np.zeros_like(u, dtype=float)
    u_abs = np.abs(u)
    mask = u_abs <= c
    rho[mask] = (c**2 / 6) * (1 - (1 - (u_abs[mask]/c)**2)**3)
    rho[~mask] = c**2 / 6
    return rho


def get_rho_huber(u: np.ndarray, c: float = 1.345) -> np.ndarray:
    """Huber ρ函数

    huber的w(u) = min(1, c/|u|)
    对应的ρ(u) = u^2/2 for |u|<=c, c|u|-c^2/2 for |u|>c

    Parameters
    ----------
    u : np.ndarray
        标准化残差
    c : float
        调节参数，默认1.345（95%效率）

    Returns
    -------
    np.ndarray
        ρ(u) 值
    """
    rho = np.zeros_like(u, dtype=float)
    u_abs = np.abs(u)
    mask = u_abs <= c
    rho[mask] = 0.5 * u_abs[mask]**2
    rho[~mask] = c * u_abs[~mask] - 0.5 * c**2
    return rho


def solve_m_scale_bisquare(resids: np.ndarray, tol: float = 1e-6, max_iter: int = 50) -> float:
    """求解M-scale方程（bisquare）

    M-scale定义: 找到σ使得 (1/n) * Σρ(r_i/σ) = δ
    对于bisquare, δ = 0.199 对应50% breakdown point

    这是S估计的核心：S估计找到使M-scale最小的参数

    Parameters
    ----------
    resids : np.ndarray
        残差向量
    tol : float
        收敛容差
    max_iter : int
        最大迭代次数

    Returns
    -------
    float
        M-scale估计值σ
    """
    n = len(resids)
    sigma = sampleMAD0(resids)  # 初始估计，使用已有函数

    if sigma < eps():  # 使用已有函数
        return sigma

    delta = 0.199  # bisquare的breakdown constant
    c = 4.685       # bisquare的调节参数

    for iteration in range(max_iter):
        u = resids / sigma
        rho_values = get_rho_bisquare(u, c)
        mean_rho = np.mean(rho_values)

        if mean_rho < eps():
            break

        # 迭代更新尺度: σ_new = σ * sqrt(mean(ρ) / δ)
        sigma_new = sigma * np.sqrt(mean_rho / delta)

        # 检查收敛
        if abs(sigma_new - sigma) < tol * sigma:
            return sigma_new

        sigma = sigma_new

    return sigma


def solve_m_scale_huber(resids: np.ndarray, tol: float = 1e-6, max_iter: int = 50) -> float:
    """求解M-scale方程（huber）

    对于huber, δ = 0.5

    Parameters
    ----------
    resids : np.ndarray
        残差向量
    tol : float
        收敛容差
    max_iter : int
        最大迭代次数

    Returns
    -------
    float
        M-scale估计值σ
    """
    n = len(resids)
    sigma = sampleMAD0(resids)

    if sigma < eps():
        return sigma

    delta = 0.5    # huber的breakdown constant
    c = 1.345      # huber的调节参数

    for iteration in range(max_iter):
        u = resids / sigma
        rho_values = get_rho_huber(u, c)
        mean_rho = np.mean(rho_values)

        if mean_rho < eps():
            break

        sigma_new = sigma * np.sqrt(mean_rho / delta)

        if abs(sigma_new - sigma) < tol * sigma:
            return sigma_new

        sigma = sigma_new

    return sigma