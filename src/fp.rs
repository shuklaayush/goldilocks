use crate::util::{add_no_canonicalize_trashing_input, branch_hint, split, sqrt_tonelli_shanks};
use crate::util::{assume, try_inverse_u64};
use core::ops::{Add, AddAssign, Mul, MulAssign, Neg, Sub, SubAssign};
use ff::{Field, PrimeField};
use rand_core::RngCore;
use std::convert::TryInto;
use std::fmt::{Debug, Formatter};
use subtle::{Choice, ConditionallySelectable, ConstantTimeEq, CtOption};

/// Goldilocks field with modulus 2^64 - 2^32 + 1.
/// A Goldilocks field may store a non-canonical form of the element
/// where the value can be between 0 and 2^64.
/// For unique representation of its form, use `to_canonical_u64`
#[derive(Clone, Copy, Default, Eq)]
pub struct Goldilocks(pub u64);

/// 2^64 - 2^32 + 1
pub const MODULUS: u64 = 0xffffffff00000001;

/// Constant representing the multiplicative generator of the modulus.
/// It's derived with SageMath with: `GF(MODULUS).primitive_element()`.
const MULTIPLICATIVE_GENERATOR: Goldilocks = Goldilocks(0x7);

/// 2^32 - 1
pub const EPSILON: u64 = 0xffffffff;

/// Constant representing the modolus as static str
const MODULUS_STR: &str = "0xFFFFFFFF00000001";

const TWO_INV: Goldilocks = Goldilocks(0x7FFFFFFF80000001);

const ZETA: Goldilocks = Goldilocks(0xFFFFFFFE00000001);

const DELTA: Goldilocks = Goldilocks(0xAA5B2509F86BB4D4);

const ROOT_OF_UNITY: Goldilocks = Goldilocks(0x185629dcda58878c);

const ROOT_OF_UNITY_INV: Goldilocks = Goldilocks(0x76B6B635B6FC8719);

impl PartialEq for Goldilocks {
    fn eq(&self, other: &Goldilocks) -> bool {
        self.to_canonical_u64() == other.to_canonical_u64()
    }
}

impl Debug for Goldilocks {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result<(), std::fmt::Error> {
        write!(f, "0x")?;
        for &b in self.0.to_be_bytes().iter() {
            write!(f, "{:02x}", b)?;
        }
        Ok(())
    }
}

impl Goldilocks {
    pub const fn zero() -> Self {
        Self(0)
    }

    pub const fn one() -> Self {
        Self(1)
    }

    // TODO: Remove?
    /// Converts a 512-bit little endian integer into
    /// a `$field` by reducing by the modulus.
    fn from_bytes_wide(bytes: &[u8; 64]) -> Goldilocks {
        let v = u128::from_le_bytes(bytes[..16].try_into().unwrap());
        Self::from_u128(v)
    }

    fn get_lower_128(&self) -> u128 {
        self.0 as u128
    }
}

impl Field for Goldilocks {
    const ZERO: Self = Self::zero();
    const ONE: Self = Self::one();

    /// Returns an element chosen uniformly at random using a user-provided RNG.
    /// Note: this sampler is not constant time!
    fn random(mut rng: impl RngCore) -> Self {
        let mut res = rng.next_u64();
        while res >= MODULUS {
            res = rng.next_u64();
        }
        Self(res)
    }

    /// Squares this element.
    #[must_use]
    fn square(&self) -> Self {
        *self * *self
    }

    /// Cubes this element.
    #[must_use]
    fn cube(&self) -> Self {
        self.square() * self
    }

    /// Doubles this element.
    #[must_use]
    fn double(&self) -> Self {
        *self + *self
    }

    /// Computes the multiplicative inverse of this element,
    /// failing if the element is zero.
    fn invert(&self) -> CtOption<Self> {
        match try_inverse_u64(&self.0) {
            Some(p) => CtOption::new(Self(p), Choice::from(1)),
            None => CtOption::new(Self(0), Choice::from(0)),
        }
    }

    /// Returns the square root of the field element, if it is
    /// quadratic residue.
    fn sqrt(&self) -> CtOption<Self> {
        // TODO
        /// `(t - 1) // 2` where t * 2^s + 1 = p with t odd.
        const T_MINUS1_OVER2: [u64; 4] = [2147483647, 0, 0, 0];
        sqrt_tonelli_shanks(self, T_MINUS1_OVER2[0])
    }

    fn sqrt_ratio(num: &Self, div: &Self) -> (Choice, Self) {
        ff::helpers::sqrt_ratio_generic(num, div)
    }
}

impl From<bool> for Goldilocks {
    fn from(bit: bool) -> Goldilocks {
        if bit {
            Goldilocks::one()
        } else {
            Goldilocks::zero()
        }
    }
}

impl core::cmp::Ord for Goldilocks {
    fn cmp(&self, other: &Self) -> core::cmp::Ordering {
        let left = self.to_repr();
        let right = other.to_repr();
        match left.0.cmp(&right.0) {
            core::cmp::Ordering::Equal => None,
            res => Some(res),
        }
        .unwrap_or(core::cmp::Ordering::Equal)
    }
}

impl core::cmp::PartialOrd for Goldilocks {
    fn partial_cmp(&self, other: &Self) -> Option<core::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl AsRef<u64> for Goldilocks {
    fn as_ref(&self) -> &u64 {
        &self.0
    }
}

impl AsMut<[u8]> for Goldilocks {
    fn as_mut(&mut self) -> &mut [u8] {
        let ptr = self as *mut Self as *mut u8;
        unsafe { std::slice::from_raw_parts_mut(ptr, 8) }
    }
}

impl AsRef<[u8]> for Goldilocks {
    fn as_ref(&self) -> &[u8] {
        let ptr = self as *const Self as *const u8;
        // SAFETY Self is repr(transparent) and u64 is 8 bytes wide,
        // with alignment greater than that of u8
        unsafe { std::slice::from_raw_parts(ptr, 8) }
    }
}

/// This represents an element of a prime field.
impl PrimeField for Goldilocks {
    /// The prime field can be converted back and forth into this binary
    /// representation.
    type Repr = Self;

    const MODULUS: &'static str = MODULUS_STR;
    const MULTIPLICATIVE_GENERATOR: Self = MULTIPLICATIVE_GENERATOR;
    const TWO_INV: Self = TWO_INV;
    /// Returns the `2^s` root of unity.
    ///
    /// It can be calculated by exponentiating `Self::multiplicative_generator` by `t`,
    /// where `t = (modulus - 1) >> Self::S`.
    const ROOT_OF_UNITY: Self = ROOT_OF_UNITY;
    const ROOT_OF_UNITY_INV: Self = ROOT_OF_UNITY_INV;
    const DELTA: Self = DELTA;
    /// How many bits are needed to represent an element of this field.
    const NUM_BITS: u32 = 64;
    /// How many bits of information can be reliably stored in the field element.
    ///
    /// This is usually `Self::NUM_BITS - 1`.
    const CAPACITY: u32 = 63;
    /// An integer `s` satisfying the equation `2^s * t = modulus - 1` with `t` odd.
    ///
    /// This is the number of leading zero bits in the little-endian bit representation of
    /// `modulus - 1`.
    const S: u32 = 32;

    /// Attempts to convert a byte representation of a field element into an element of
    /// this prime field, failing if the input is not canonical (is not smaller than the
    /// field's modulus).
    ///
    /// The byte representation is interpreted with the same endianness as elements
    /// returned by [`PrimeField::to_repr`].
    fn from_repr(repr: Self::Repr) -> CtOption<Self> {
        CtOption::new(repr, Choice::from(1))
    }

    /// Attempts to convert a byte representation of a field element into an element of
    /// this prime field, failing if the input is not canonical (is not smaller than the
    /// field's modulus).
    ///
    /// The byte representation is interpreted with the same endianness as elements
    /// returned by [`PrimeField::to_repr`].
    ///
    /// # Security
    ///
    /// This method provides **no** constant-time guarantees. Implementors of the
    /// `PrimeField` trait **may** optimise this method using non-constant-time logic.
    fn from_repr_vartime(repr: Self::Repr) -> Option<Self> {
        Self::from_repr(repr).into()
    }

    /// Converts an element of the prime field into the standard byte representation for
    /// this field.
    ///
    /// The endianness of the byte representation is implementation-specific. Generic
    /// encodings of field elements should be treated as opaque.
    fn to_repr(&self) -> Self::Repr {
        *self
    }

    /// Returns true iff this element is odd.
    fn is_odd(&self) -> Choice {
        Choice::from((self.0 & 1) as u8)
    }

    fn from_u128(v: u128) -> Self {
        Self(reduce128(v).to_canonical_u64())
    }
}

impl From<u64> for Goldilocks {
    fn from(input: u64) -> Self {
        Self(input)
    }
}

impl From<Goldilocks> for u64 {
    fn from(input: Goldilocks) -> Self {
        input.0
    }
}

impl ConditionallySelectable for Goldilocks {
    fn conditional_select(a: &Self, b: &Self, choice: Choice) -> Self {
        Self(u64::conditional_select(&a.0, &b.0, choice))
    }
}

impl ConstantTimeEq for Goldilocks {
    fn ct_eq(&self, other: &Self) -> Choice {
        self.to_canonical_u64().ct_eq(&other.to_canonical_u64())
    }
}

impl Neg for Goldilocks {
    type Output = Self;

    #[inline]
    fn neg(self) -> Self {
        if self.0 == 0 {
            self
        } else {
            Self(MODULUS - self.to_canonical_u64())
        }
    }
}

impl Add for Goldilocks {
    type Output = Self;

    #[inline]
    #[allow(clippy::suspicious_arithmetic_impl)]
    fn add(self, rhs: Self) -> Self::Output {
        let (sum, over) = self.0.overflowing_add(rhs.0);
        let (mut sum, over) = sum.overflowing_add((over as u64) * EPSILON);
        if over {
            // NB: self.0 > Self::ORDER && rhs.0 > Self::ORDER is necessary but not sufficient for
            // double-overflow.
            // This assume does two things:
            //  1. If compiler knows that either self.0 or rhs.0 <= ORDER, then it can skip this
            //     check.
            //  2. Hints to the compiler how rare this double-overflow is (thus handled better with
            //     a branch).
            assume(self.0 > MODULUS && rhs.0 > MODULUS);
            branch_hint();
            sum += EPSILON; // Cannot overflow.
        }
        Self(sum)
    }
}

impl<'a> Add<&'a Goldilocks> for Goldilocks {
    type Output = Self;

    #[inline]
    fn add(self, rhs: &'a Goldilocks) -> Self::Output {
        self + *rhs
    }
}

impl AddAssign for Goldilocks {
    #[inline]
    fn add_assign(&mut self, rhs: Self) {
        *self = *self + rhs;
    }
}

impl<'a> AddAssign<&'a Goldilocks> for Goldilocks {
    #[inline]
    fn add_assign(&mut self, rhs: &'a Goldilocks) {
        *self = *self + *rhs;
    }
}

impl Sub for Goldilocks {
    type Output = Self;

    #[inline]
    #[allow(clippy::suspicious_arithmetic_impl)]
    fn sub(self, rhs: Self) -> Self {
        let (diff, under) = self.0.overflowing_sub(rhs.0);
        let (mut diff, under) = diff.overflowing_sub((under as u64) * EPSILON);
        if under {
            // NB: self.0 < EPSILON - 1 && rhs.0 > Self::ORDER is necessary but not sufficient for
            // double-underflow.
            // This assume does two things:
            //  1. If compiler knows that either self.0 >= EPSILON - 1 or rhs.0 <= ORDER, then it
            //     can skip this check.
            //  2. Hints to the compiler how rare this double-underflow is (thus handled better
            //     with a branch).
            assume(self.0 < EPSILON - 1 && rhs.0 > MODULUS);
            branch_hint();
            diff -= EPSILON; // Cannot underflow.
        }
        Self(diff)
    }
}

impl<'a> Sub<&'a Goldilocks> for Goldilocks {
    type Output = Self;

    #[inline]
    fn sub(self, rhs: &'a Goldilocks) -> Self::Output {
        self - *rhs
    }
}

impl SubAssign for Goldilocks {
    #[inline]
    fn sub_assign(&mut self, rhs: Self) {
        *self = *self - rhs;
    }
}

impl<'a> SubAssign<&'a Goldilocks> for Goldilocks {
    #[inline]
    fn sub_assign(&mut self, rhs: &'a Goldilocks) {
        *self = *self - *rhs;
    }
}

impl Mul for Goldilocks {
    type Output = Self;

    #[inline]
    fn mul(self, rhs: Self) -> Self {
        reduce128((self.0 as u128) * (rhs.0 as u128))
    }
}

impl<'a> Mul<&'a Goldilocks> for Goldilocks {
    type Output = Self;

    #[inline]
    fn mul(self, rhs: &'a Goldilocks) -> Self::Output {
        self * *rhs
    }
}

impl MulAssign for Goldilocks {
    #[inline]
    fn mul_assign(&mut self, rhs: Self) {
        *self = *self * rhs;
    }
}

impl<'a> MulAssign<&'a Goldilocks> for Goldilocks {
    #[inline]
    fn mul_assign(&mut self, rhs: &'a Goldilocks) {
        *self = *self * *rhs;
    }
}

impl<T: ::core::borrow::Borrow<Goldilocks>> ::core::iter::Sum<T> for Goldilocks {
    fn sum<I: Iterator<Item = T>>(iter: I) -> Self {
        iter.fold(Self::ZERO, |acc, item| acc + item.borrow())
    }
}

impl<T: ::core::borrow::Borrow<Goldilocks>> ::core::iter::Product<T> for Goldilocks {
    fn product<I: Iterator<Item = T>>(iter: I) -> Self {
        iter.fold(Self::ONE, |acc, item| acc * item.borrow())
    }
}

/// Reduces to a 64-bit value. The result might not be in canonical form; it could be in between the
/// field order and `2^64`.
#[inline]
fn reduce128(x: u128) -> Goldilocks {
    let (x_lo, x_hi) = split(x); // This is a no-op
    let x_hi_hi = x_hi >> 32;
    let x_hi_lo = x_hi & EPSILON;

    let (mut t0, borrow) = x_lo.overflowing_sub(x_hi_hi);
    if borrow {
        branch_hint(); // A borrow is exceedingly rare. It is faster to branch.
        t0 -= EPSILON; // Cannot underflow.
    }
    let t1 = x_hi_lo * EPSILON;
    let t2 = unsafe { add_no_canonicalize_trashing_input(t0, t1) };
    Goldilocks(t2)
}

impl Goldilocks {
    #[inline]
    pub fn to_canonical_u64(&self) -> u64 {
        let mut c = self.0;
        // We only need one condition subtraction, since 2 * ORDER would not fit in a u64.
        if c >= MODULUS {
            c -= MODULUS;
        }
        c
    }

    pub const fn size() -> usize {
        8
    }

    pub fn legendre(&self) -> LegendreSymbol {
        // s = self^((modulus - 1) // 2)
        // 9223372034707292160
        let s = 0x7fffffff80000000;
        let s = self.pow_vartime(&[s]);
        if s == Self::zero() {
            LegendreSymbol::Zero
        } else if s == Self::one() {
            LegendreSymbol::QuadraticResidue
        } else {
            LegendreSymbol::QuadraticNonResidue
        }
    }

    pub fn batch_multiplicative_inverse(x: &[Self]) -> Vec<Self> {
        // This is Montgomery's trick. At a high level, we invert the product of the given field
        // elements, then derive the individual inverses from that via multiplication.

        // The usual Montgomery trick involves calculating an array of cumulative products,
        // resulting in a long dependency chain. To increase instruction-level parallelism, we
        // compute WIDTH separate cumulative product arrays that only meet at the end.

        // Higher WIDTH increases instruction-level parallelism, but too high a value will cause us
        // to run out of registers.
        const WIDTH: usize = 4;
        // JN note: WIDTH is 4. The code is specialized to this value and will need
        // modification if it is changed. I tried to make it more generic, but Rust's const
        // generics are not yet good enough.

        // Handle special cases. Paradoxically, below is repetitive but concise.
        // The branches should be very predictable.
        let n = x.len();
        if n == 0 {
            return Vec::new();
        } else if n == 1 {
            return vec![x[0].invert().unwrap()];
        } else if n == 2 {
            let x01 = x[0] * x[1];
            let x01inv = x01.invert().unwrap();
            return vec![x01inv * x[1], x01inv * x[0]];
        } else if n == 3 {
            let x01 = x[0] * x[1];
            let x012 = x01 * x[2];
            let x012inv = x012.invert().unwrap();
            let x01inv = x012inv * x[2];
            return vec![x01inv * x[1], x01inv * x[0], x012inv * x01];
        }
        debug_assert!(n >= WIDTH);

        // Buf is reused for a few things to save allocations.
        // Fill buf with cumulative product of x, only taking every 4th value. Concretely, buf will
        // be [
        //   x[0], x[1], x[2], x[3],
        //   x[0] * x[4], x[1] * x[5], x[2] * x[6], x[3] * x[7],
        //   x[0] * x[4] * x[8], x[1] * x[5] * x[9], x[2] * x[6] * x[10], x[3] * x[7] * x[11],
        //   ...
        // ].
        // If n is not a multiple of WIDTH, the result is truncated from the end. For example,
        // for n == 5, we get [x[0], x[1], x[2], x[3], x[0] * x[4]].
        let mut buf: Vec<Self> = Vec::with_capacity(n);
        // cumul_prod holds the last WIDTH elements of buf. This is redundant, but it's how we
        // convince LLVM to keep the values in the registers.
        let mut cumul_prod: [Self; WIDTH] = x[..WIDTH].try_into().unwrap();
        buf.extend(cumul_prod);
        for (i, &xi) in x[WIDTH..].iter().enumerate() {
            cumul_prod[i % WIDTH] *= xi;
            buf.push(cumul_prod[i % WIDTH]);
        }
        debug_assert_eq!(buf.len(), n);

        let mut a_inv = {
            // This is where the four dependency chains meet.
            // Take the last four elements of buf and invert them all.
            let c01 = cumul_prod[0] * cumul_prod[1];
            let c23 = cumul_prod[2] * cumul_prod[3];
            let c0123 = c01 * c23;
            let c0123inv = c0123.invert().unwrap();
            let c01inv = c0123inv * c23;
            let c23inv = c0123inv * c01;
            [
                c01inv * cumul_prod[1],
                c01inv * cumul_prod[0],
                c23inv * cumul_prod[3],
                c23inv * cumul_prod[2],
            ]
        };

        for i in (WIDTH..n).rev() {
            // buf[i - WIDTH] has not been written to by this loop, so it equals
            // x[i % WIDTH] * x[i % WIDTH + WIDTH] * ... * x[i - WIDTH].
            buf[i] = buf[i - WIDTH] * a_inv[i % WIDTH];
            // buf[i] now holds the inverse of x[i].
            a_inv[i % WIDTH] *= x[i];
        }
        for i in (0..WIDTH).rev() {
            buf[i] = a_inv[i];
        }

        for (&bi, &xi) in buf.iter().zip(x) {
            // Sanity check only.
            debug_assert_eq!(bi * xi, Self::one());
        }

        buf
    }
}

#[derive(Debug, PartialEq, Eq)]
pub enum LegendreSymbol {
    Zero = 0,
    QuadraticResidue = 1,
    QuadraticNonResidue = -1,
}
