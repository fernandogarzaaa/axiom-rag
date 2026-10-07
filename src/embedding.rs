//! Deterministic hash-based text embeddings (feature hashing).
//!
//! No model download, no API key, works fully offline. Each token is hashed
//! into a fixed-size vector with a random sign (the "hashing trick",
//! Weinberger et al., 2009): the sign makes unrelated tokens cancel out in
//! expectation, while shared tokens accumulate. Vectors are L2-normalized, so
//! cosine similarity is just the dot product.
//!
//! This is a deliberately simple baseline. It captures lexical overlap well
//! and degrades gracefully on paraphrase; the agentic retrieval loop in
//! [`crate::retrieval`] compensates by reformulating queries when scores are
//! low.

/// Deterministic embedder. Same text always yields the same vector, in every
/// process and on every machine.
pub struct Embedder {
    dim: usize,
}

impl Embedder {
    /// Create an embedder with `dim` dimensions (`dim` must be > 0).
    pub fn new(dim: usize) -> Self {
        assert!(dim > 0, "embedding dimension must be positive");
        Self { dim }
    }

    /// Embedding dimension.
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// Embed `text` into an L2-normalized vector.
    pub fn embed(&self, text: &str) -> Vec<f32> {
        let mut v = vec![0f32; self.dim];
        for tok in tokenize(text) {
            let h1 = fnv1a_64(tok.as_bytes(), 0xcbf29ce484222325);
            let h2 = fnv1a_64(tok.as_bytes(), 0x84222325cbf29ce4);
            let idx = (h1 % self.dim as u64) as usize;
            let sign = if h2 & 1 == 0 { 1.0 } else { -1.0 };
            v[idx] += sign;
        }
        let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 0.0 {
            for x in &mut v {
                *x /= norm;
            }
        }
        v
    }

    /// Cosine similarity of two (assumed normalized) vectors.
    pub fn cosine(a: &[f32], b: &[f32]) -> f32 {
        debug_assert_eq!(a.len(), b.len());
        a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
    }
}

/// Lowercase alphanumeric tokens of length >= 2.
pub fn tokenize(text: &str) -> Vec<String> {
    text.split(|c: char| !c.is_alphanumeric())
        .filter(|s| s.len() >= 2)
        .map(|s| s.to_lowercase())
        .collect()
}

/// FNV-1a 64-bit with an explicit seed. `std`'s `DefaultHasher` (SipHash) uses
/// per-process random keys, so it is NOT stable across runs; this is.
fn fnv1a_64(data: &[u8], seed: u64) -> u64 {
    let mut h = seed;
    for b in data {
        h ^= *b as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    h
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deterministic_across_calls() {
        let e = Embedder::new(512);
        assert_eq!(e.embed("hello world"), e.embed("hello world"));
    }

    #[test]
    fn output_is_unit_norm() {
        let e = Embedder::new(256);
        let v = e.embed("the quick brown fox jumps over the lazy dog");
        let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-5, "norm was {norm}");
    }

    #[test]
    fn self_similarity_is_one() {
        let e = Embedder::new(512);
        let v = e.embed("rust systems programming");
        let s = Embedder::cosine(&v, &v);
        assert!((s - 1.0).abs() < 1e-5, "self similarity was {s}");
    }

    #[test]
    fn shared_tokens_score_higher_than_unrelated() {
        let e = Embedder::new(512);
        let q = e.embed("rust borrow checker ownership");
        let related = e.embed("ownership and borrowing in rust");
        let unrelated = e.embed("photosynthesis chlorophyll sunlight");
        assert!(Embedder::cosine(&q, &related) > Embedder::cosine(&q, &unrelated));
    }

    #[test]
    fn empty_text_gives_zero_vector() {
        let e = Embedder::new(64);
        let v = e.embed("");
        assert!(v.iter().all(|&x| x == 0.0));
    }

    #[test]
    fn tokenize_filters_short_tokens() {
        let toks = tokenize("a bb ccc 123!");
        assert_eq!(toks, vec!["bb", "ccc", "123"]);
    }
}
