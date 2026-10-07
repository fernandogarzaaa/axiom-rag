//! Verifier-gated answer generation.
//!
//! Generation is extractive on purpose: the answer is composed of sentences
//! taken verbatim from retrieved chunks, each paired with a citation. An
//! extractive answer cannot hallucinate new facts; the only remaining risk is
//! picking an irrelevant sentence, which [`crate::verifier::verify_answer`]
//! guards against. If nothing survives verification, the system abstains
//! instead of answering.

use std::collections::HashSet;

use crate::chunking::Chunk;
use crate::config::RagConfig;
use crate::embedding::Embedder;
use crate::retrieval::fingerprint;
use crate::verifier::claim_supported;

/// A source backing one answer sentence.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct Citation {
    /// 1-based number as shown in the answer text.
    pub number: usize,
    /// Index of the sentence in [`Answer::sentences`] this cites.
    pub sentence_index: usize,
    /// Source document of the chunk.
    pub source: String,
    /// Chunk index within the source.
    pub chunk_index: usize,
    /// The chunk text (so verification needs no store handle).
    pub chunk_text: String,
}

/// A generated answer with per-sentence citations.
#[derive(Debug, Clone)]
pub struct Answer {
    /// Sentences joined with citation markers, e.g. `"... [1] ... [2]"`.
    pub text: String,
    /// The raw sentences.
    pub sentences: Vec<String>,
    /// One citation per sentence, in order.
    pub citations: Vec<Citation>,
    /// True when the system abstained for lack of supported evidence.
    pub abstained: bool,
}

/// Split text into sentences.
///
/// The text is first split into blocks on blank lines (so markdown headers
/// and list items do not glue into one pseudo-sentence), then each block is
/// split on `.`, `!`, `?` followed by whitespace or end of input. Sentences
/// shorter than 4 tokens are dropped as fragments.
pub fn split_sentences(text: &str) -> Vec<String> {
    let mut out = Vec::new();
    for block in text.split("\n\n") {
        let block = block.trim();
        if block.is_empty() {
            continue;
        }
        let mut start = 0;
        let bytes = block.as_bytes();
        let mut i = 0;
        while i < bytes.len() {
            let c = bytes[i] as char;
            if (c == '.' || c == '!' || c == '?')
                && (i + 1 == bytes.len() || (bytes[i + 1] as char).is_whitespace())
            {
                push_sentence(&mut out, &block[start..=i]);
                start = i + 1;
            }
            i += 1;
        }
        push_sentence(&mut out, &block[start..]);
    }
    out
}

fn push_sentence(out: &mut Vec<String>, s: &str) {
    let s = s.trim().replace('\n', " ");
    let s = s.split_whitespace().collect::<Vec<_>>().join(" ");
    // Skip headers and labels: a segment ending in ':' is never a complete claim.
    if s.ends_with(':') {
        return;
    }
    if s.split_whitespace().count() >= 4 {
        out.push(s);
    }
}

/// Build a cited answer from retrieved `(chunk, score)` pairs.
///
/// Selection is greedy by query similarity; each candidate sentence must be
/// supported by at least one chunk at `claim_threshold` before it is emitted.
/// A second, more lenient pass runs if the first yields nothing; persistent
/// failure produces an explicit abstention rather than an unverified answer.
pub fn generate_answer(
    question: &str,
    hits: &[(Chunk, f32)],
    embedder: &Embedder,
    config: &RagConfig,
) -> Answer {
    for pass in 0..2 {
        let threshold = if pass == 0 {
            config.claim_threshold
        } else {
            config.claim_threshold * 0.5
        };
        let widen = if pass == 0 { 1.0 } else { 1.5 };
        if let Some(answer) = try_generate(question, hits, embedder, config, threshold, widen) {
            return answer;
        }
    }
    Answer {
        text:
            "I couldn't find supported evidence in the indexed documents to answer this question."
                .to_string(),
        sentences: Vec::new(),
        citations: Vec::new(),
        abstained: true,
    }
}

fn try_generate(
    question: &str,
    hits: &[(Chunk, f32)],
    embedder: &Embedder,
    config: &RagConfig,
    threshold: f64,
    widen: f64,
) -> Option<Answer> {
    if hits.is_empty() {
        return None;
    }
    let q_emb = embedder.embed(question);

    // Candidate sentences with their best supporting chunk.
    let mut candidates: Vec<(f32, String, usize)> = Vec::new(); // (score, sentence, chunk_pos)
    let mut seen: HashSet<String> = HashSet::new();
    for (pos, (chunk, _)) in hits.iter().enumerate() {
        for sent in split_sentences(&chunk.text) {
            let fp = fingerprint(&sent);
            if !seen.insert(fp) {
                continue;
            }
            let s_emb = embedder.embed(&sent);
            let score = Embedder::cosine(&q_emb, &s_emb);
            // The sentence must be supported by at least one retrieved chunk.
            let best_chunk = hits
                .iter()
                .enumerate()
                .filter(|(_, (c, _))| claim_supported(&sent, &c.text, embedder, threshold))
                .max_by(|a, b| {
                    let sa = Embedder::cosine(&s_emb, &embedder.embed(&a.1 .0.text));
                    let sb = Embedder::cosine(&s_emb, &embedder.embed(&b.1 .0.text));
                    sa.partial_cmp(&sb).unwrap_or(std::cmp::Ordering::Equal)
                })
                .map(|(p, _)| p)
                .unwrap_or(pos);
            if claim_supported(&sent, &hits[best_chunk].0.text, embedder, threshold) {
                candidates.push((score * widen as f32, sent, best_chunk));
            }
        }
    }
    if candidates.is_empty() {
        return None;
    }
    candidates.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

    let mut sentences = Vec::new();
    let mut citations = Vec::new();
    for (i, (_, sent, chunk_pos)) in candidates
        .into_iter()
        .take(config.max_answer_sentences.max(1))
        .enumerate()
    {
        let (chunk, _) = &hits[chunk_pos];
        sentences.push(sent);
        citations.push(Citation {
            number: i + 1,
            sentence_index: i,
            source: chunk.source.clone(),
            chunk_index: chunk.index,
            chunk_text: chunk.text.clone(),
        });
    }
    let text = sentences
        .iter()
        .enumerate()
        .map(|(i, s)| format!("{s} [{}]", i + 1))
        .collect::<Vec<_>>()
        .join(" ");
    Some(Answer {
        text,
        sentences,
        citations,
        abstained: false,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn chunk(text: &str, source: &str, index: usize) -> Chunk {
        Chunk {
            text: text.to_string(),
            source: source.to_string(),
            index,
            byte_start: 0,
            byte_end: text.len(),
        }
    }

    #[test]
    fn split_sentences_basic() {
        let v = split_sentences(
            "Hello world foo bar. Second sentence here now! Short. Final one ok yes.",
        );
        assert_eq!(v.len(), 3);
        assert!(v[0].ends_with('.'));
    }

    #[test]
    fn generates_cited_answer_from_hits() {
        let e = Embedder::new(256);
        let config = RagConfig::default();
        let hits = vec![
            (
                chunk(
                    "The Axiom engine performs test-time training on device. It needs no cloud connection at all.",
                    "a.md",
                    0,
                ),
                0.5,
            ),
            (
                chunk(
                    "Photosynthesis converts sunlight into chemical energy inside plant leaves every day.",
                    "b.md",
                    0,
                ),
                0.1,
            ),
        ];
        let ans = generate_answer("What does the Axiom engine do?", &hits, &e, &config);
        assert!(!ans.abstained);
        assert!(!ans.sentences.is_empty());
        assert_eq!(ans.sentences.len(), ans.citations.len());
        assert!(ans.text.contains("[1]"));
        // every citation points at a real chunk
        for c in &ans.citations {
            assert!(hits
                .iter()
                .any(|(ch, _)| ch.source == c.source && ch.index == c.chunk_index));
        }
    }

    #[test]
    fn abstains_when_no_sentences_extractable() {
        let e = Embedder::new(256);
        let config = RagConfig::default();
        // All fragments are under the 4-token sentence minimum, so no
        // candidate sentences exist and the answer must abstain.
        let hits = vec![(chunk("a b c. d e f. g h i!", "g.md", 0), 0.9)];
        let ans = generate_answer("anything at all here", &hits, &e, &config);
        assert!(ans.abstained);
        assert!(ans.sentences.is_empty());
        assert!(ans.text.contains("couldn't find supported evidence"));
    }

    #[test]
    fn empty_hits_abstains() {
        let e = Embedder::new(256);
        let ans = generate_answer("anything", &[], &e, &RagConfig::default());
        assert!(ans.abstained);
    }
}
