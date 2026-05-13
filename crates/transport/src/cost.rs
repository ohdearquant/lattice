//! Cost matrix abstractions.
//!
//! The central engineering choice is to separate **how** transport cost is
//! computed from **how** Sinkhorn uses it. This keeps the solver generic over
//! dense matrices, on-the-fly pairwise distances, and custom callback-based
//! costs without changing the numerical kernel.

use core::fmt;

use super::math::sqrt;

/// Errors that arise while constructing or validating cost structures.
///
/// **Stable** (provisional): error type paired with the `Stable` cost API; changes only if the cost trait surface changes.
#[derive(Debug, Clone, PartialEq)]
pub enum CostError {
    /// Point set has no elements.
    EmptyPointSet,
    /// Flat buffer length is not divisible by the stated dimension.
    InvalidContiguousLayout {
        /// Total number of scalar values in the flat buffer.
        values: usize,
        /// Stated embedding dimension.
        dim: usize,
    },
    /// A point has the wrong number of dimensions.
    DimensionMismatch {
        /// Number of dimensions expected for all points in this set.
        expected: usize,
        /// Actual number of dimensions found for the offending point.
        found: usize,
        /// Zero-based index of the offending point.
        index: usize,
    },
    /// A point coordinate is NaN or infinite.
    NonFinitePointValue {
        /// Zero-based index of the point containing the non-finite coordinate.
        point: usize,
        /// Zero-based dimension index of the non-finite coordinate.
        dim: usize,
        /// The non-finite value that was encountered.
        value: f32,
    },
}

impl fmt::Display for CostError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyPointSet => write!(f, "point set is empty"),
            Self::InvalidContiguousLayout { values, dim } => write!(
                f,
                "contiguous point storage with {values} values is not divisible by dimension {dim}"
            ),
            Self::DimensionMismatch {
                expected,
                found,
                index,
            } => write!(
                f,
                "point {index} has dimension {found}, expected {expected}"
            ),
            Self::NonFinitePointValue { point, dim, value } => write!(
                f,
                "point {point} dimension {dim} contains non-finite value {value}"
            ),
        }
    }
}

/// Common interface for accessing points without committing to a storage layout.
///
/// **Stable** (provisional): core cost abstraction; signature frozen pending second consumer.
pub trait PointSet {
    /// Number of points in the set.
    fn len(&self) -> usize;
    /// Dimensionality of each point.
    fn dim(&self) -> usize;
    /// Access point at index `idx`.
    fn point(&self, idx: usize) -> &[f32];
    /// Whether the set is empty.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

impl PointSet for [Vec<f32>] {
    fn len(&self) -> usize {
        <[Vec<f32>]>::len(self)
    }

    fn dim(&self) -> usize {
        self.first().map_or(0, Vec::len)
    }

    fn point(&self, idx: usize) -> &[f32] {
        &self[idx]
    }
}

impl<'a> PointSet for [&'a [f32]] {
    fn len(&self) -> usize {
        <[&'a [f32]]>::len(self)
    }

    fn dim(&self) -> usize {
        self.first().map_or(0, |row| row.len())
    }

    fn point(&self, idx: usize) -> &[f32] {
        self[idx]
    }
}

/// Contiguous row-major storage for point clouds.
///
/// Convenient when embeddings already live in a flat buffer from a BLAS,
/// database page, or GPU staging area.
///
/// **Stable** (provisional): lightweight view type; layout unlikely to change.
#[derive(Debug, Clone, Copy)]
pub struct ContiguousPoints<'a> {
    data: &'a [f32],
    dim: usize,
}

impl<'a> ContiguousPoints<'a> {
    /// Create from a flat buffer with the given dimension per point.
    ///
    /// **Stable** (provisional): constructor matches struct stability.
    pub fn new(data: &'a [f32], dim: usize) -> Result<Self, CostError> {
        if dim == 0 || data.is_empty() {
            return Err(CostError::EmptyPointSet);
        }
        if data.len() % dim != 0 {
            return Err(CostError::InvalidContiguousLayout {
                values: data.len(),
                dim,
            });
        }
        Ok(Self { data, dim })
    }
}

impl PointSet for ContiguousPoints<'_> {
    fn len(&self) -> usize {
        self.data.len() / self.dim
    }

    fn dim(&self) -> usize {
        self.dim
    }

    fn point(&self, idx: usize) -> &[f32] {
        let start = idx * self.dim;
        let end = start + self.dim;
        &self.data[start..end]
    }
}

/// Distance function between two points.
///
/// **Stable** (provisional): `Copy` bound and two-method signature are stable; custom metrics implement this.
pub trait PointMetric: Copy {
    /// Compute the distance (cost) between two points.
    fn distance(&self, lhs: &[f32], rhs: &[f32]) -> f32;
    /// Human-readable name for this metric.
    fn name(&self) -> &'static str {
        "custom"
    }
}

/// Squared Euclidean cost `||x - y||^2`, appropriate for Wasserstein-2.
///
/// **Stable** (provisional): standard metric; no breaking change anticipated.
#[derive(Debug, Clone, Copy, Default)]
pub struct SquaredEuclidean;

impl PointMetric for SquaredEuclidean {
    fn distance(&self, lhs: &[f32], rhs: &[f32]) -> f32 {
        if lhs.len() != rhs.len() {
            return f32::NAN;
        }
        lhs.iter()
            .zip(rhs.iter())
            .map(|(&x, &y)| {
                let delta = x - y;
                delta * delta
            })
            .sum()
    }

    fn name(&self) -> &'static str {
        "squared_euclidean"
    }
}

/// Cosine distance `1 - cos(theta)`.
///
/// If embeddings are already unit-normalized, set `assume_unit_norm = true` to
/// avoid recomputing norms for every pair.
///
/// **Stable** (provisional): standard metric; `norm_floor` field may gain a default-constructor helper but struct shape is stable.
#[derive(Debug, Clone, Copy)]
pub struct CosineDistance {
    /// Skip norm computation for pre-normalized vectors.
    pub assume_unit_norm: bool,
    /// Floor for norm values to avoid division by zero.
    pub norm_floor: f32,
}

impl Default for CosineDistance {
    fn default() -> Self {
        Self {
            assume_unit_norm: false,
            norm_floor: 1e-12,
        }
    }
}

impl PointMetric for CosineDistance {
    fn distance(&self, lhs: &[f32], rhs: &[f32]) -> f32 {
        if lhs.len() != rhs.len() {
            return f32::NAN;
        }
        let cosine = if self.assume_unit_norm {
            let mut dot = 0.0f32;
            for (&x, &y) in lhs.iter().zip(rhs.iter()) {
                dot += x * y;
            }
            dot
        } else {
            let mut dot = 0.0f32;
            let mut lhs_norm_sq = 0.0f32;
            let mut rhs_norm_sq = 0.0f32;
            for (&x, &y) in lhs.iter().zip(rhs.iter()) {
                dot += x * y;
                lhs_norm_sq += x * x;
                rhs_norm_sq += y * y;
            }
            let lhs_norm = sqrt(lhs_norm_sq.max(0.0));
            let rhs_norm = sqrt(rhs_norm_sq.max(0.0));
            dot / (lhs_norm.max(self.norm_floor) * rhs_norm.max(self.norm_floor))
        };
        (1.0 - cosine).clamp(0.0, 2.0)
    }

    fn name(&self) -> &'static str {
        "cosine"
    }
}

/// Generic cost matrix interface.
///
/// **Stable** (provisional): three-method trait; the core abstraction boundary between geometry and solver.
pub trait CostMatrix {
    /// Number of source entries (rows).
    fn rows(&self) -> usize;
    /// Number of target entries (columns).
    fn cols(&self) -> usize;
    /// Cost of transporting from source `row` to target `col`.
    fn cost(&self, row: usize, col: usize) -> f32;
}

/// Dense row-major cost matrix.
///
/// **Stable** (provisional): primary cost matrix type used throughout the crate.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct DenseCostMatrix {
    rows: usize,
    cols: usize,
    data: Vec<f32>,
}

impl DenseCostMatrix {
    /// Create from pre-computed data. Panics in debug mode if dimensions mismatch.
    ///
    /// **Stable** (provisional): constructor matches struct stability.
    pub fn new(rows: usize, cols: usize, data: Vec<f32>) -> Self {
        // FP-030: validate shape in release builds too — a caller-provided
        // dimension mismatch corrupts every subsequent cost() lookup.
        assert_eq!(
            rows * cols,
            data.len(),
            "DenseCostMatrix: expected {} elements ({}×{}), got {}",
            rows * cols,
            rows,
            cols,
            data.len()
        );
        Self { rows, cols, data }
    }

    /// Build a cost matrix by evaluating a closure for each (row, col) pair.
    ///
    /// **Stable** (provisional): convenience constructor; signature stable.
    pub fn from_fn<F>(rows: usize, cols: usize, mut f: F) -> Self
    where
        F: FnMut(usize, usize) -> f32,
    {
        let mut data = Vec::with_capacity(rows * cols);
        for row in 0..rows {
            for col in 0..cols {
                data.push(f(row, col));
            }
        }
        Self { rows, cols, data }
    }

    /// Build from two point sets and a distance metric.
    ///
    /// **Stable** (provisional): primary way to turn point cloud pairs into a cost matrix.
    pub fn from_point_sets<X, Y, M>(source: &X, target: &Y, metric: M) -> Result<Self, CostError>
    where
        X: PointSet + ?Sized,
        Y: PointSet + ?Sized,
        M: PointMetric,
    {
        validate_point_set(source)?;
        validate_point_set(target)?;
        Ok(Self::from_fn(source.len(), target.len(), |row, col| {
            metric.distance(source.point(row), target.point(col))
        }))
    }

    /// Access the underlying flat data.
    ///
    /// **Stable** (provisional): zero-copy accessor; no breaking change anticipated.
    #[inline]
    pub fn as_slice(&self) -> &[f32] {
        &self.data
    }

    /// Compute the flat index for (row, col).
    ///
    /// **Unstable**: layout-specific helper; may be removed if storage changes.
    #[inline]
    pub fn index(&self, row: usize, col: usize) -> usize {
        row * self.cols + col
    }
}

impl CostMatrix for DenseCostMatrix {
    fn rows(&self) -> usize {
        self.rows
    }

    fn cols(&self) -> usize {
        self.cols
    }

    fn cost(&self, row: usize, col: usize) -> f32 {
        self.data[self.index(row, col)]
    }
}

/// Pairwise cost computed lazily from two point clouds and a metric.
///
/// **Stable** (provisional): lazy cost view; avoids materializing a dense matrix for large problems.
#[derive(Debug, Clone, Copy)]
pub struct PairwiseCostMatrix<'a, X: ?Sized, Y: ?Sized, M> {
    /// Source point set.
    pub source: &'a X,
    /// Target point set.
    pub target: &'a Y,
    /// Distance metric.
    pub metric: M,
}

impl<'a, X: ?Sized, Y: ?Sized, M> PairwiseCostMatrix<'a, X, Y, M> {
    /// Create a lazy pairwise cost matrix.
    ///
    /// **Stable** (provisional): constructor matches struct stability.
    pub fn new(source: &'a X, target: &'a Y, metric: M) -> Self {
        Self {
            source,
            target,
            metric,
        }
    }
}

impl<X, Y, M> CostMatrix for PairwiseCostMatrix<'_, X, Y, M>
where
    X: PointSet + ?Sized,
    Y: PointSet + ?Sized,
    M: PointMetric,
{
    fn rows(&self) -> usize {
        self.source.len()
    }

    fn cols(&self) -> usize {
        self.target.len()
    }

    fn cost(&self, row: usize, col: usize) -> f32 {
        self.metric
            .distance(self.source.point(row), self.target.point(col))
    }
}

/// Arbitrary callback-backed cost matrix.
///
/// Useful for non-Euclidean domain-specific penalties, for example
/// hybrid symbolic/neural costs or graph-aware memory migration penalties.
///
/// **Stable** (provisional): escape hatch for custom cost functions; closure type is generic so the public shape won't change.
#[derive(Clone, Copy)]
pub struct ClosureCostMatrix<F> {
    rows: usize,
    cols: usize,
    f: F,
}

impl<F> fmt::Debug for ClosureCostMatrix<F> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ClosureCostMatrix")
            .field("rows", &self.rows)
            .field("cols", &self.cols)
            .finish_non_exhaustive()
    }
}

impl<F> ClosureCostMatrix<F> {
    /// Create a closure-backed cost matrix.
    ///
    /// **Stable** (provisional): constructor matches struct stability.
    pub fn new(rows: usize, cols: usize, f: F) -> Self {
        Self { rows, cols, f }
    }
}

impl<F> CostMatrix for ClosureCostMatrix<F>
where
    F: Fn(usize, usize) -> f32,
{
    fn rows(&self) -> usize {
        self.rows
    }

    fn cols(&self) -> usize {
        self.cols
    }

    fn cost(&self, row: usize, col: usize) -> f32 {
        (self.f)(row, col)
    }
}

/// Validates that a point set is non-empty, has constant dimension, and does
/// not contain NaNs or infinities.
///
/// **Unstable**: internal validation helper; may be inlined or split as the API evolves.
pub fn validate_point_set<P>(points: &P) -> Result<(), CostError>
where
    P: PointSet + ?Sized,
{
    if points.is_empty() || points.dim() == 0 {
        return Err(CostError::EmptyPointSet);
    }
    let dim = points.dim();
    for point_idx in 0..points.len() {
        let point = points.point(point_idx);
        if point.len() != dim {
            return Err(CostError::DimensionMismatch {
                expected: dim,
                found: point.len(),
                index: point_idx,
            });
        }
        for (dim_idx, &value) in point.iter().enumerate() {
            if !value.is_finite() {
                return Err(CostError::NonFinitePointValue {
                    point: point_idx,
                    dim: dim_idx,
                    value,
                });
            }
        }
    }
    Ok(())
}
