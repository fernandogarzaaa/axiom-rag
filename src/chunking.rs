//! Sliding-window chunking over words with configurable overlap.
//!
//! Word-based (not character-based) windows keep chunks aligned to token
//! boundaries, which matters for the hash embeddings in [`crate::embedding`]:
//! a chunk that cuts a word in half would hash differently from the same
//! content chunked cleanly.

use serde::{Deserialize, Serialize};

/// One chunk of a source document.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Chunk {
    /// Chunk text.
    pub text: String,
    /// Source identifier (usually the file path).
    pub source: String,
    /// Zero-based chunk number within the source.
    pub index: usize,
    /// Byte offset of the chunk start in the source text.
    pub byte_start: usize,
    /// Byte offset of the chunk end in the source text.
    pub byte_end: usize,
}

/// Split `text` into overlapping word windows.
///
/// * `chunk_size_words`: words per window (minimum 1).
/// * `overlap_words`: words shared with the next window (clamped below the
///   window size so the window always advances).
///
/// Returns an empty vec for empty/whitespace-only input.
pub fn chunk_text(
    text: &str,
    source: &str,
    chunk_size_words: usize,
    overlap_words: usize,
) -> Vec<Chunk> {
    // Word spans as byte offsets so chunk boundaries never split a char.
    let mut words: Vec<(usize, usize)> = Vec::new();
    let mut start: Option<usize> = None;
    for (i, c) in text.char_indices() {
        if c.is_whitespace() {
            if let Some(s) = start.take() {
                words.push((s, i));
            }
        } else if start.is_none() {
            start = Some(i);
        }
    }
    if let Some(s) = start {
        words.push((s, text.len()));
    }
    if words.is_empty() {
        return Vec::new();
    }

    let size = chunk_size_words.max(1);
    let overlap = overlap_words.min(size - 1);
    let step = (size - overlap).max(1);

    let mut chunks = Vec::new();
    let mut index = 0;
    let mut w = 0;
    while w < words.len() {
        let end = (w + size).min(words.len());
        let (bs, _) = words[w];
        let (_, be) = words[end - 1];
        chunks.push(Chunk {
            text: text[bs..be].to_string(),
            source: source.to_string(),
            index,
            byte_start: bs,
            byte_end: be,
        });
        index += 1;
        if end == words.len() {
            break;
        }
        w += step;
    }
    chunks
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_text_yields_no_chunks() {
        assert!(chunk_text("", "s", 10, 2).is_empty());
        assert!(chunk_text("   \n\t ", "s", 10, 2).is_empty());
    }

    #[test]
    fn short_text_yields_single_chunk() {
        let chunks = chunk_text("hello world", "s", 200, 50);
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].text, "hello world");
        assert_eq!(chunks[0].index, 0);
    }

    #[test]
    fn sliding_window_counts() {
        // 10 words, window 4, overlap 1 -> step 3: [0..4], [3..7], [6..10]
        let text = "a b c d e f g h i j";
        let chunks = chunk_text(text, "s", 4, 1);
        assert_eq!(chunks.len(), 3);
        assert_eq!(chunks[0].text, "a b c d");
        assert_eq!(chunks[1].text, "d e f g");
        assert_eq!(chunks[2].text, "g h i j");
    }

    #[test]
    fn overlap_clamped_below_window_size() {
        let text = "a b c d e f";
        // overlap >= size would never advance; it is clamped to size-1.
        let chunks = chunk_text(text, "s", 2, 99);
        assert!(chunks.len() > 1);
        // step is 1, windows: [0..2],[1..3],[2..4],[3..5],[4..6]
        assert_eq!(chunks.len(), 5);
    }

    #[test]
    fn byte_offsets_are_valid() {
        let text = "héllo wörld foo bar";
        let chunks = chunk_text(text, "s", 2, 0);
        for c in &chunks {
            assert!(text.get(c.byte_start..c.byte_end).is_some());
            assert_eq!(&text[c.byte_start..c.byte_end], c.text);
        }
    }

    #[test]
    fn consecutive_chunks_share_overlap_words() {
        let text: String = (0..20)
            .map(|i| format!("w{i}"))
            .collect::<Vec<_>>()
            .join(" ");
        let chunks = chunk_text(&text, "s", 6, 2);
        assert!(chunks.len() >= 2);
        let last_words: Vec<&str> = chunks[0].text.split_whitespace().collect();
        let first_words: Vec<&str> = chunks[1].text.split_whitespace().collect();
        assert_eq!(&last_words[last_words.len() - 2..], &first_words[..2]);
    }
}
